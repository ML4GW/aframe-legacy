import itertools
import logging
import time
from concurrent.futures import FIRST_EXCEPTION, Future, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from lal import gpstime
from mldatafind.authenticate import authenticate
from mldatafind.io import fetch_timeseries
from mldatafind.segments import query_segments
from typeo import scriptify

from bbhnet.io import h5
from bbhnet.io.timeslides import TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import AsyncExecutor

# TODO: Integrate updated inference format once that PR is merged

logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


@dataclass
class Shift:
    ifos: List[str]
    shifts: Iterable[float]

    def __post_init__(self):
        self.shifts = [float(i) for i in self.shifts]
        self._i = 0

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i >= len(self.ifos):
            raise StopIteration

        ifo, shift = self.ifos[self._i], self.shifts[self._i]
        self._i += 1
        return ifo, shift

    def __str__(self):
        return "-".join([f"{i[0]}{j}" for i, j in zip(self.ifos, self.shifts)])


def intify(x: float):
    return int(x) if int(x) == x else x


def make_shifts(
    ifos: Iterable[str], shifts: Iterable[float], n_slides: int
) -> List[Shift]:
    ranges = [range(n_slides) for i in shifts if i]
    shift_objs = []
    for rng in itertools.product(*ranges):
        it = iter(rng)
        shift = []
        for i in shifts:
            shift.append(0 if i == 0 else next(it) * i)
        shift = Shift(ifos, shift)
        shift_objs.append(shift)

    return shift_objs


def submit_write(
    pool: AsyncExecutor, ts: TimeSlide, t: np.ndarray, **fields: np.ndarray
) -> Future:
    ts_type = ts.path.name
    if ts_type == "background":
        prefix = "raw"
    else:
        prefix = "inj"

    future = pool.submit(
        h5.write_timeseries,
        ts.path,
        prefix=prefix,
        t=t,
        **fields,
    )

    future.add_done_callback(
        lambda f: logging.debug(f"Wrote background {ts_type} {f.result()}")
    )
    return future


def check_segment(
    shifts: List[Shift],
    datadir: Path,
    segment_start: float,
    dur: float,
    min_segment_length: Optional[float] = None,
    force_generation: bool = False,
):
    # first check if we'll even have enough data for
    # this segment to be worth working with
    if min_segment_length is not None and dur < min_segment_length:
        return None

    segment_start = intify(segment_start)
    dur = intify(dur)

    # then check if _all_ data for this segment
    # exists in each shift separately
    fields, prefixes = ["background", "injection"], ["raw", "inj"]
    segment_shifts = []
    for shift in shifts:
        for field, prefix in zip(fields, prefixes):
            dirname = datadir / f"dt-{shift}" / field
            fname = f"{prefix}_{segment_start}-{dur}.hdf5"
            if not (dirname / fname).exists() or force_generation:
                # we don't have data for this segment at this
                # shift value, so we'll need to create it
                segment_shifts.append(shift)
                break

    return segment_shifts


def find_segments(
    flags,
    start: float,
    min_segment_length: float,
    min_livetime: float,
    wait: int = 120,  # length of time dqsegdb takes to update
):
    """
    Returns segments that will be used for evaluation
    """
    segments = []
    while True:
        stop = int(gpstime.gps_time_now())
        stop = 1262696622
        if stop - start < min_segment_length:
            time.sleep(stop - start)

        segments = query_segments(
            flags,
            start,
            stop,
            min_segment_length,
        )
        livetime = sum([stop - start for start, stop in segments])
        if livetime > min_livetime:
            return segments
        logging.info(
            f"Only {livetime} seconds of livetime available, waiting for more"
        )
        time.sleep(wait)


def calc_shifts_required(
    segments: List[Tuple[int, int]], Tb: float, shift: float
):
    """
    Based off of the lengths of the segments and the
    amount of data that will need to be sloughed off
    the ends due to shifting, calculate how many shifts
    will be required to achieve Tb seconds worth of background

    Args:
        segments: A list of tuples of the start and stop times of the segments
        Tb: The amount of background data to generate
        shift: The increment to shift the data by

    Returns the number of shifts required to achieve Tb seconds of background
    """

    livetime = sum([stop - start for start, stop in segments])
    n_segments = len(segments)
    shifts_required = 0
    while True:
        max_shift = shift * shifts_required
        total_livetime = (livetime - n_segments * max_shift) * shifts_required
        if total_livetime < Tb:
            shifts_required += 1
            continue
        break

    return shifts_required


@scriptify
def main(
    logdir: Path,
    datadir: Path,
    ifos: List[str],
    channel: str,
    flag: str,
    Tb: float,
    sample_rate: float,
    shifts: Iterable[float],
    min_segment_length: float,
    min_livetime: float,
    verbose: bool = True,
) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    configure_logging(logdir / "evaluation.log", verbose)

    start = int(gpstime.gps_time_now())
    start = 1262686622
    flags = [f"{ifo}:{flag}" for ifo in ifos]
    channels = [f"{ifo}:{channel}" for ifo in ifos]

    authenticate()
    segments = find_segments(
        flags,
        start,
        min_segment_length,
        min_livetime,
    )

    shifts_required = calc_shifts_required(segments, Tb, max(shifts))
    max_shift = max(shifts) * shifts_required

    # Need to exclude the zero-lag data
    shifts = make_shifts(ifos, shifts, shifts_required + 1)
    shifts = shifts[1:]

    with AsyncExecutor(4, thread=False) as pool:
        for segment_start, segment_stop in segments:
            dur = segment_stop - segment_start - max_shift
            seg_str = f"{segment_start}-{segment_stop}"

            segment_shifts = check_segment(
                shifts,
                datadir,
                segment_start,
                dur,
                min_segment_length,
            )

            if segment_shifts is None:
                logging.info(f"Segment {seg_str} too short, skipping")
                continue
            elif len(segment_shifts) == 0:
                logging.info(
                    f"All data for segment {seg_str} already exists, skipping"
                )
                continue

            # begin the download of data in a separate thread
            logging.debug(f"Beginning download of segment {seg_str}")
            background = fetch_timeseries(
                channels,
                segment_start,
                segment_stop,
            )
            logging.debug(f"Completed download of segment {seg_str}")
            background = background.resample(sample_rate)

            # set up array of times for all shifts

            times = np.arange(
                segment_start, segment_start + dur, 1 / sample_rate
            )

            futures = []

            for shift in segment_shifts:
                logging.debug(
                    "Creating timeslide for segment {} "
                    "with shifts {}".format(seg_str, shift)
                )

                # 1. start by creating all the directories we'll need
                root = datadir / f"dt-{shift}"
                root.mkdir(exist_ok=True, parents=True)

                raw_ts = TimeSlide.create(root=root, field="background")

                # 2. Then create the appropriate shifts for each
                # interferometer and save them to their raw
                # directory

                # time array is always relative to first shift value
                background_data = {}
                for (ifo, shift_val), channel in zip(shift, channels):
                    start = segment_start + shift_val
                    bckgrd = background[channel].crop(start, start + dur)
                    background_data[ifo] = bckgrd.value

                future = submit_write(pool, raw_ts, times, **background_data)
                futures.append(future)

            wait(futures, return_when=FIRST_EXCEPTION)


if __name__ == "__main__":
    main()
