import glob
import logging
import os
from typing import Optional

import h5py
import numpy as np
from gwpy.segments import Segment, SegmentList
from gwpy.timeseries import TimeSeries
from hermes.typeo import typeo
from tqdm import tqdm

"""
Tools to generate a dataset of glitches from omicron triggers.

For information on how the omicron triggers were generated see:

/home/ethan.marx/bbhnet/generate-glitch-dataset/omicron/12566/H1L1_1256665618_100000
/runfiles/omicron_params_H1.txt

on CIT cluster for an example omicron parameter file.

Of note is the clustering timescale of 1 second

"""


def veto(times: list, segmentlist: SegmentList):

    """
    Remove events from a list of times based on a segmentlist
    A time ``t`` will be vetoed if ``start <= t <= end`` for any veto
    segment in the list.

    Arguments:
    - times: the times of event triggers to veto
    - segmentliss: the list of veto segments to use

    Returns:
    - keep_bools: list of booleans; True for the triggers to keep
    """

    # find args that sort times and create sorted times array
    sorted_args = np.argsort(times)
    sorted_times = times[sorted_args]

    # initiate array of args to keep;
    # refers to original args of unsorted times array;
    # begin with all args being kept

    keep_bools = np.ones(times.shape[0], dtype=bool)

    # initiate loop variables; extract first segment
    j = 0
    a, b = segmentlist[j]
    i = 0

    while i < sorted_times.size:
        t = sorted_times[i]

        # if before start, not in vetoed segment; move to next trigger now
        if t < a:

            # original arg is the ith sorted arg
            i += 1
            continue

        # if after end, find the next segment and check this trigger again
        if t > b:
            j += 1
            try:
                a, b = segmentlist[j]
                continue
            except IndexError:
                break

        # otherwise it must be in veto segment; move on to next trigger
        original_arg = sorted_args[i]
        keep_bools[original_arg] = False
        i += 1

    return keep_bools


def generate_glitch_dataset(
    ifo: str,
    snr_thresh: float,
    start: float,
    stop: float,
    window: float,
    sample_rate: float,
    trig_file: str,
    vetoes: SegmentList = None,
):

    """
    Generates a list of omicron trigger times that satisfy snr threshold

    Arguments:
    - ifo: ifo to generate glitch triggers for
    - snr_thresh: snr threshold above which to keep as glitch
    - start: start gpstime
    - stop: stop gpstime
    - window: half window around trigger time to query data for
    - sample_rate: sampling frequency
    - trig_file: txt file output from omicron triggers
            (first column is gps times, 3rd column is snrs)
    - vetoes: SegmentList object of times to ignore
    """

    glitches = []
    snrs = []

    # snr and time columns in omicron file
    snr_col = 2
    time_col = 0

    # load in triggers
    triggers = np.loadtxt(trig_file)

    # if passed, apply vetos
    if vetoes is not None:
        keep_bools = veto(triggers[:, time_col], vetoes)
        triggers = triggers[keep_bools]

    # restrict triggers to within gps start and stop times
    times = triggers[:, time_col]
    time_args = np.logical_and(times > start, times < stop)
    triggers = triggers[time_args]

    # apply snr thresh
    day_snrs = triggers[:, snr_col]  # second column is snrs
    snr_thresh_args = np.where(day_snrs > snr_thresh)
    triggers = triggers[snr_thresh_args]

    logging.info(f"Querying data for {len(triggers)} triggers")

    # query data for each trigger
    for trigger in tqdm(triggers):
        time = trigger[time_col]
        print(time)
        try:
            glitch_ts = TimeSeries.fetch_open_data(
                ifo, time - window, time + window
            )

            glitch_ts = glitch_ts.resample(sample_rate)

            snrs.append(trigger[snr_col])
            glitches.append(glitch_ts)

        except ValueError:
            logging.info(f"Data not available for trigger at time: {time}")
            continue

    glitches = np.array(glitches)
    snrs = np.array(snrs)
    return glitches, snrs


@typeo
def main(
    snr_thresh: float,
    start: float,
    stop: float,
    window: float,
    omicron_dir: str,
    out_dir: str,
    sample_rate: float = 4096,
    H1_veto_file: Optional[str] = None,
    L1_veto_file: Optional[str] = None,
):

    """Simulates a set of glitches for both
        H1 and L1 that can be added to background

    Arguments:

    - snr_thresh: snr threshold above which to keep as glitch
    - start: start gpstime
    - stop: stop gpstime
    - window: half window around trigger time to query data for
    - sample_rate: sampling frequency
    - out_dir: output directory to which signals will be written
    - omicron_dir: base directory of omicron triggers
            (see /home/ethan.marx/bbhnet/generate-glitch-dataset/omicron/)
    - H1_veto_file: path to file containing vetoes for H1
    - L1_veto_file: path to file containing vetoes for L1
    """

    # create logging file in model_dir
    logging.basicConfig(
        filename=f"{out_dir}/log.log",
        format="%(message)s",
        filemode="w",
        level=logging.INFO,
    )

    # if passed, load in H1 vetoes and convert to gwpy SegmentList object
    if H1_veto_file is not None:

        logging.info("Applying vetoes to H1 times")
        logging.info(f"H1 veto file: {H1_veto_file}")

        # load in H1 vetoes
        H1_vetoes = np.loadtxt(H1_veto_file)

        # convert arrays to gwpy Segment objects
        H1_vetoes = [Segment(seg[0], seg[1]) for seg in H1_vetoes]

        # create SegmentList object
        H1_vetoes = SegmentList(H1_vetoes).coalesce()
    else:
        H1_vetoes = None

    if L1_veto_file is not None:
        logging.info("Applying vetoes to L1 times")
        logging.info(f"L1 veto file: {L1_veto_file}")

        L1_vetoes = np.loadtxt(L1_veto_file)
        L1_vetoes = [Segment(seg[0], seg[1]) for seg in L1_vetoes]
        L1_vetoes = SegmentList(L1_vetoes).coalesce()
    else:
        L1_vetoes = None

    # omicron triggers are split up by directories
    # into segments of 10^5 seconds
    # get paths for relevant directories
    # based on start and stop gpstimes passed by user

    gps_day_start = str(start)[:5]
    gps_day_end = str(stop)[:5]
    all_gps_days = np.arange(int(gps_day_start), int(gps_day_end) + 1, 1)

    H1_glitches = []
    L1_glitches = []

    H1_snrs = []
    L1_snrs = []

    # loop over gps days
    for i, day in enumerate(all_gps_days):

        # get path for this gps day
        omicron_path = os.path.join(
            omicron_dir, f"{day}/*/PostProc/unclustered/"
        )

        # the path to the omicron triggers
        H1_trig_file = glob.glob(
            os.path.join(omicron_path, "triggers_unclustered_H1.txt")
        )[0]
        L1_trig_file = glob.glob(
            os.path.join(omicron_path, "triggers_unclustered_L1.txt")
        )[0]

        H1_day_glitches, H1_day_snrs = generate_glitch_dataset(
            "H1",
            snr_thresh,
            start,
            stop,
            window,
            sample_rate,
            H1_trig_file,
            vetoes=H1_vetoes,
        )

        L1_day_glitches, L1_day_snrs = generate_glitch_dataset(
            "L1",
            snr_thresh,
            start,
            stop,
            window,
            sample_rate,
            L1_trig_file,
            vetoes=L1_vetoes,
        )

        # concat
        H1_glitches.append(H1_day_glitches)
        L1_glitches.append(L1_day_glitches)

        H1_snrs.append(H1_day_snrs)
        L1_snrs.append(L1_day_snrs)

    glitch_file = os.path.join(out_dir, "glitches.h5")

    with h5py.File(glitch_file, "w") as f:
        f.create_dataset("H1_glitches", data=H1_glitches)
        f.create_dataset("H1_snrs", data=H1_snrs)

        f.create_dataset("L1_glitches", data=L1_glitches)
        f.create_dataset("L1_snrs", data=L1_snrs)

    return glitch_file


if __name__ == "__main__":
    main()
