"""
Script that generates a dataset of glitches from omicron triggers.
"""

import configparser
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from gwpy.timeseries import TimeSeries
from omicron.cli.process import main as omicron_main
from typeo import scriptify

from aframe.logging import configure_logging

# re for omicron trigger files
re_trigger_fname = re.compile(
    r"(?P<ifo>[A-Za-z]1)-(?P<name>.+)-(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*).h5$"  # noqa
)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def strain_from_triggers(
    path: Path,
    pad: Tuple[float, float],
    snr_thresh: float,
    start: float,
    stop: float,
    sample_rate: float,
    channel: str,
    chunk_size: float = 4096,
):
    """
    Generates a list of omicron trigger times that satisfy snr threshold

    Args:
        snr_thresh: snr threshold above which to keep as glitch
        start: start gpstime
        stop: stop gpstime
        window: half window around trigger time to query data for
        sample_rate: sampling arequency
        channel: channel name used to read data (including ifo prefix)
        frame_type: frame type for data discovery w/ gwdatafind
        trigger_files: List of h5 files of omicron triggers
    """

    glitches = []
    snrs = []
    gpstimes = []

    with h5py.File(path) as f:
        # restrict triggers to within gps start and stop times
        # and apply snr threshold
        triggers = f["triggers"][:]
        times = triggers["time"][:]
        mask = triggers["snr"][:] > snr_thresh
        triggers = triggers[mask]

    if len(triggers) == 0:
        return
    # re-set 'start' and 'stop' so we aren't querying unnecessary data
    start = np.min(triggers["time"]) - pad[0]
    stop = np.max(triggers["time"]) + pad[1]

    logging.debug(
        f"Querying {len(triggers)} triggers from {path}"
        f" over {stop - start} seconds of data "
    )

    # TODO: debug mldatafind generator that has
    # been leading to segmentation faults
    data = TimeSeries.get(
        channel,
        start=start,
        end=stop,
    )

    # restrict to triggers within current data chunk
    data = data.resample(sample_rate)
    times = data.times.value
    mask = (triggers["time"] > times[0] + pad[0]) & (
        triggers["time"] < times[-1] - pad[1]
    )
    chunk_triggers = triggers[mask]
    # query data for each trigger
    for trigger in chunk_triggers:
        time = trigger["time"]
        try:
            glitch_ts = data.crop(time - pad[0], time + pad[1])
        except ValueError:
            logging.warning(f"Data not available for trigger at time: {time}")
            continue
        else:
            glitches.append(list(glitch_ts.value))
            snrs.append(trigger["snr"])
            gpstimes.append(time)

    glitches = np.stack(glitches)
    return glitches, snrs, gpstimes


def omicron_main_wrapper(
    start: int,
    stop: int,
    run_dir: Path,
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
    config.set(section, "frametype", f"{ifo}_{frame_type}")
    config.set(section, "channels", f"{ifo}:{channel}")
    config.set(section, "cluster-dt", str(cluster_dt))
    config.set(section, "sample-frequency", str(sample_rate))
    config.set(section, "chunk-duration", str(chunk_duration))
    config.set(section, "segment-duration", str(segment_duration))
    config.set(section, "overlap-duration", str(overlap))
    config.set(section, "mismatch-max", str(mismatch_max))
    config.set(section, "snr-threshold", str(snr_thresh))
    # in an online setting, can also pass state-vector,
    # and bits to check for science mode
    config.set(section, "state-flag", f"{ifo}:{state_flag}")

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
        "request_disk=4GB",
        "--output-dir",
        str(run_dir),
        "--skip-gzip",
        "--skip-rm",
    ]
    if verbose:
        omicron_args += ["--verbose"]

    # create and launch omicron dag
    omicron_main(omicron_args)

    # return variables necessary for glitch generation
    return ifo


def validate_cache():
    pass


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
    channel: str,
    frame_type: str,
    sample_rate: float,
    state_flag: str,
    ifos: List[str],
    fduration: float,
    psd_length: float,
    kernel_length: float,
    chunk_size: float = 4096,
    analyze_testing_set: bool = False,
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
    """

    logdir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)

    log_file = logdir / "glitches.log"
    configure_logging(log_file, verbose)

    # cached = validate_cache(datadir)
    cached = False
    if cached and not force_generation:
        logging.info(
            "Glitch data already exists and forced generation is off. "
            "Not generating glitches"
        )
        return

    # nyquist
    f_max = sample_rate / 2

    glitches = {}
    snrs = {}
    times = {}
    train_futures = []

    run_dir = datadir / "omicron"
    train_run_dir = run_dir / "training"
    test_run_dir = run_dir / "testing"
    omicron_log_file = run_dir / "pyomicron.log"

    # seconds before and after trigger time so that
    # we have enough data such that we
    # can sample the trigger time at the first or last sample of the kernel
    pad = (psd_length + kernel_length + fduration, kernel_length + fduration)

    for ifo in ifos:
        train_ifo_dir = train_run_dir / ifo
        test_ifo_dir = test_run_dir / ifo
        train_ifo_dir.mkdir(exist_ok=True, parents=True)

        # launch pyomicron futures for training set and testing sets.
        # as train futures complete, launch glitch generation processes.
        # let test set jobs run in background as glitch data is queried
        pool = ThreadPoolExecutor(4)
        args = [
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
            omicron_log_file,
            verbose,
        ]

        train_future = pool.submit(
            omicron_main_wrapper, start, stop, train_ifo_dir, *args
        )
        train_futures.append(train_future)

        if analyze_testing_set:
            test_ifo_dir.mkdir(exist_ok=True, parents=True)
            pool.submit(
                omicron_main_wrapper, stop, test_stop, test_ifo_dir, *args
            )

    for future in as_completed(train_futures):
        ifo = future.result()
        outdir = datadir / "glitches" / ifo
        outdir.mkdir(exist_ok=True, parents=True)
        trigger_dir = train_run_dir / ifo / "merge" / f"{ifo}:{channel}"
        trigger_files = sorted(list(trigger_dir.iterdir()))

        for f in trigger_files:
            match = re_trigger_fname.search(f.name)
            if match is not None:
                t0, length = match.group("t0"), match.group("length")
                logging.info(
                    f"Generating glitch dataset for {ifo} and file {f}"
                )
                out = outdir / f"{ifo}-glitches-{t0}-{length}.hdf5"

                glitches, snrs, times = strain_from_triggers(
                    f,
                    pad,
                    snr_thresh,
                    start,
                    stop,
                    sample_rate,
                    f"{ifo}:{channel}",
                    chunk_size,
                )

                with h5py.File(out, "a") as f:
                    g = f.create_group(f"{ifo}")
                    g.create_dataset("glitches", data=glitches)
                    g.create_dataset("snrs", data=snrs)
                    g.create_dataset("times", data=times)

    return datadir


if __name__ == "__main__":
    main()
