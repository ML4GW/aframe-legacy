"""
Script that generates a dataset of glitches from omicron triggers.
"""

import configparser
import logging
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
from gwpy.segments import Segment, SegmentList
from gwpy.timeseries import TimeSeries
from omicron.cli.process import main as omicron_main
from tqdm import tqdm
from typeo import scriptify

from bbhnet.logging import configure_logging


def veto(times: list, segmentlist: SegmentList):

    """
    Remove events from a list of times based on a segmentlist
    A time ``t`` will be vetoed if ``start <= t <= end`` for any veto
    segment in the list.

    Args:
    - times: the times of event triggers to veto
    - segmentlist: the list of veto segments to use

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
    snr_thresh: float,
    start: float,
    stop: float,
    window: float,
    sample_rate: float,
    channel: str,
    trig_file: str,
    vetoes: SegmentList = None,
):

    """
    Generates a list of omicron trigger times that satisfy snr threshold

    Args:
        ifo: ifo to generate glitch triggers for
        snr_thresh: snr threshold above which to keep as glitch
        start: start gpstime
        stop: stop gpstime
        window: half window around trigger time to query data for
        sample_rate: sampling arequency
        channel: channel name used to read data
        frame_type: frame type for data discovery w/ gwdatafind
        trig_file: txt file output from omicron triggers
            (first column is gps times, 3rd column is snrs)
        vetoes: SegmentList object of times to ignore
    """

    glitches = []
    snrs = []

    # load in triggers
    with h5py.File(trig_file) as f:
        triggers = f["triggers"][()]

        # restrict triggers to within gps start and stop times
        # and apply snr threshold
        times = triggers["time"][()]
        mask = (times > start) & (times < stop)
        mask &= triggers["snr"][()] > snr_thresh
        triggers = triggers[mask]

    # if passed, apply vetos
    if vetoes is not None:
        keep_bools = veto(times, vetoes)
        times = times[keep_bools]
        snrs = snrs[keep_bools]

    # re-set 'start' and 'stop' so we aren't querying unnecessary data
    start = np.min(triggers["time"]) - 2 * window
    stop = np.max(triggers["time"]) + 2 * window

    logging.info(
        f"Querying {stop - start} seconds of data for {len(triggers)} triggers"
    )

    ts = TimeSeries.get(channel, start=start, end=stop, pad=0)

    # for each trigger
    for trigger in tqdm(triggers):
        time = trigger["time"]

        try:
            glitch_ts = ts.crop(time - window, time + window)
        except ValueError:
            logging.warning(f"Data not available for trigger at time: {time}")
            continue
        else:
            glitch_ts = glitch_ts.resample(sample_rate)
            glitches.append(glitch_ts)
            snrs.append(trigger["snr"])

    glitches = np.array(glitches)
    snrs = np.array(snrs)
    return glitches, snrs


def omicron_main_wrapper(
    start: int,
    stop: int,
    q_min: float,
    q_max: float,
    f_min: float,
    f_max: float,
    sample_rate: float,
    cluster_dt: float,
    chunk_duration: int,
    segment_duration: int,
    overlap: int,
    mismatch_max: float,
    snr_thresh: float,
    frame_type: str,
    channel: str,
    state_flag: str,
    ifo: str,
    run_dir: Path,
    log_file: Path,
    verbose: bool,
):

    """Parses args into a format compatible for Pyomicron,
    then launches omicron dag
    """

    # pyomicron expects some arguments passed via
    # a config file. Create that config file

    config = configparser.ConfigParser()
    section = "GW"
    config.add_section(section)

    config.set(section, "q-range", f"{q_min} {q_max}")
    config.set(section, "frequency-range", f"{f_min} {f_max}")
    config.set(section, "frametype", f"{frame_type}")
    config.set(section, "channels", f"{channel}")
    config.set(section, "cluster-dt", str(cluster_dt))
    config.set(section, "sample-frequency", str(sample_rate))
    config.set(section, "chunk-duration", str(chunk_duration))
    config.set(section, "segment-duration", str(segment_duration))
    config.set(section, "overlap-duration", str(overlap))
    config.set(section, "mismatch-max", str(mismatch_max))
    config.set(section, "snr-threshold", str(snr_thresh))

    config.add_section("OUTPUTS")
    config.set("OUTPUTS", "format", "hdf5")

    # in an online setting, can also pass state-vector,
    # and bits to check for science mode
    config.set(section, "state-flag", f"{state_flag}")

    config_file_path = run_dir / f"omicron_{ifo}.ini"

    # write config file
    with open(config_file_path, "w") as configfile:
        config.write(configfile)

    # parse args into format expected by omicron
    omicron_args = [
        section,
        "--log-file",
        str(log_file),
        "--config-file",
        str(config_file_path),
        "--gps",
        f"{start}",
        f"{stop}",
        "--ifo",
        ifo,
        "-c",
        "request_disk=100",
        "--output-dir",
        str(run_dir),
        "--skip-gzip",
        "--skip-rm",
    ]
    if verbose:
        omicron_args += ["--verbose"]

    # create and launch omicron dag
    omicron_main(omicron_args)


@scriptify
def main(
    snr_thresh: float,
    start: int,
    stop: int,
    test_stop: int,
    q_min: float,
    q_max: float,
    f_min: float,
    cluster_dt: float,
    chunk_duration: int,
    segment_duration: int,
    overlap: int,
    mismatch_max: float,
    window: float,
    datadir: Path,
    logdir: Path,
    channels: Iterable[str],
    frame_types: Iterable[str],
    sample_rate: float,
    state_flags: Iterable[str],
    veto_files: Optional[dict[str, str]] = None,
    force_generation: bool = False,
    verbose: bool = False,
):

    """Generates a set of glitches for both
        H1 and L1 that can be added to background

        First, an omicron job is launched via pyomicron
        (https://github.com/gwpy/pyomicron/). Next, triggers (i.e. glitches)
        above a given SNR threshold are selected, and data is queried
        for these triggers and saved in an h5 file.

    Arguments:
        snr_thresh: snr threshold above which to keep as glitch
        start: start gpstime
        stop: training stop gpstime
        test_stop: testing stop gpstime
        q_min: minimum q value of tiles for omicron
        q_max: maximum q value of tiles for omicron
        f_min: lowest frequency for omicron to consider
        cluster_dt: time window for omicron to cluster neighboring triggers
        chunk_duration: duration of data (seconds) for PSD estimation
        segment_duration: duration of data (seconds) for FFT
        overlap: overlap (seconds) between neighbouring segments and chunks
        mismatch_max: maximum distance between (Q, f) tiles
        window: half window around trigger time to query data for
        sample_rate: sampling frequency
        outdir: output directory to which signals will be written
        channel: channel name used to read data
        frame_type: frame type for data discovery w/ gwdatafind
        sample_rate: sampling frequency of timeseries data
        state_flag: identifier for which segments to use
        ifos: which ifos to generate glitches for
        veto_files:
            dictionary where key is ifo and value is path
            to file containing vetoes
    """

    logdir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)

    log_file = logdir / "generate_glitches.log"
    configure_logging(log_file, verbose)

    # output file
    glitch_file = datadir / "glitches.h5"

    if glitch_file.exists() and not force_generation:
        logging.info(
            "Glitch data already exists and forced generation is off. "
            "Not generating glitches"
        )
        return

    # nyquist
    f_max = sample_rate / 2

    ifos = [channel.split(":")[0] for channel in channels]
    data_zip = zip(ifos, channels, frame_types, state_flags)

    glitches = {}
    snrs = {}
    run_dir = datadir / "omicron"

    for ifo, channel, frame_type, state_flag in data_zip:
        train_run_dir = run_dir / "training" / ifo
        test_run_dir = run_dir / "testing" / ifo

        train_run_dir.mkdir(exist_ok=True, parents=True)
        test_run_dir.mkdir(exist_ok=True, parents=True)

        # launch omicron dag for training set
        omicron_main_wrapper(
            start,
            stop,
            q_min,
            q_max,
            f_min,
            f_max,
            sample_rate,
            cluster_dt,
            chunk_duration,
            segment_duration,
            overlap,
            mismatch_max,
            snr_thresh,
            frame_type,
            channel,
            state_flag,
            ifo,
            train_run_dir,
            log_file,
            verbose,
        )

        # launch omicron dag for testing set
        # we currently don't use this information in the pipeline
        omicron_main_wrapper(
            stop,
            test_stop,
            q_min,
            q_max,
            f_min,
            f_max,
            sample_rate,
            cluster_dt,
            chunk_duration,
            segment_duration,
            overlap,
            mismatch_max,
            snr_thresh,
            frame_type,
            channel,
            state_flag,
            ifo,
            test_run_dir,
            log_file,
            verbose,
        )

        # load in vetoes and convert to gwpy SegmentList object
        if veto_files is not None:
            veto_file = veto_files[ifo]

            logging.info(f"Applying vetoes to {ifo} times")

            # load in vetoes
            vetoes = np.loadtxt(veto_file)

            # convert arrays to gwpy Segment objects
            vetoes = [Segment(seg[0], seg[1]) for seg in vetoes]

            # create SegmentList object
            vetoes = SegmentList(vetoes).coalesce()
        else:
            vetoes = None

        # get the path to the omicron triggers from *training* set
        # only use the first segment for training (should only be one)
        trigger_dir = train_run_dir / "merge" / channel
        trigger_file = sorted(list(trigger_dir.glob("*.h5")))[0]

        # generate glitches and store
        glitches[ifo], snrs[ifo] = generate_glitch_dataset(
            snr_thresh,
            start,
            stop,
            window,
            sample_rate,
            channel,
            trigger_file,
            vetoes=vetoes,
        )

        if np.isnan(glitches[ifo]).any():
            raise ValueError("The glitch data contains NaN values")

    # store glitches from training set
    with h5py.File(glitch_file, "w") as f:
        for ifo in ifos:
            f.create_dataset(f"{ifo}_glitches", data=glitches[ifo])
            f.create_dataset(f"{ifo}_snrs", data=snrs[ifo])

    return glitch_file


if __name__ == "__main__":
    main()
