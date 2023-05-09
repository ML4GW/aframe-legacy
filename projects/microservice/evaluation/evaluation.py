import logging
import time
from concurrent.futures import FIRST_EXCEPTION, wait
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from datagen.utils.timeslides import check_segment, make_shifts, submit_write
from lal import gpstime
from mldatafind.io import fetch_timeseries
from mldatafind.segments import query_segments
from typeo import scriptify

from bbhnet.io.timeslides import TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import AsyncExecutor


def find_segments(
    segment_names: List[str],
    start: float,
    min_segment_length: float,
    min_livetime: float,
    wait: int = 120,  # length of time dqsegdb takes to update
):
    """
    Returns a segment that will be used for evaluation
    """
    segments = []
    while True:
        stop = int(gpstime.gps_time_now())
        segments = query_segments(
            segment_names,
            start,
            stop,
            min_segment_length,
        )
        livetime = sum([stop - start for start, stop in segments])
        if livetime > min_livetime:
            return segments

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
    segment_names: List[str],
    ifos: List[str],
    channels: List[str],
    Tb: float,
    sample_rate: float,
    shifts: Iterable[float],
    min_segment_length: float,
    min_livetime: float,
    verbose: bool = False,
) -> None:

    logdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    configure_logging(logdir / "timeslide_injections.log", verbose)

    start = int(gpstime.gps_time_now())
    segments = find_segments(
        segment_names,
        start,
        min_segment_length,
        min_livetime,
    )

    shifts_required = calc_shifts_required(segments, Tb, max(shifts))
    max_shift = max(shifts) * shifts_required

    shifts = make_shifts(ifos, shifts, shifts_required)

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
                for ifo, shift_val in shift:
                    start = segment_start + shift_val
                    bckgrd = background[ifo].crop(start, start + dur)
                    background_data[ifo] = bckgrd.value

                future = submit_write(pool, raw_ts, times, **background_data)
                futures.append(future)

            wait(futures, return_when=FIRST_EXCEPTION)
