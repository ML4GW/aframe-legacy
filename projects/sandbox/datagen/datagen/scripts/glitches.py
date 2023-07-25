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
from mldatafind.authenticate import authenticate
from omicron.cli.process import main as omicron_main
from typeo import scriptify

from aframe.deploy import condor
from aframe.logging import configure_logging

# re for omicron trigger files
re_trigger_fname = re.compile(
    r"(?P<ifo>[A-Za-z]1)-(?P<name>.+)-(?P<t0>\d{10}\.*\d*)-(?P<length>\d+\.*\d*).h5$"  # noqa
)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def intify(x: float):
    return int(x) if int(x) == x else x


@scriptify
def collect_glitches(
    trigger_path: Path,
    outfile: Path,
    pad: Tuple[float, float],
    snr_thresh: float,
    start: float,
    stop: float,
    sample_rate: float,
    channel: str,
):
    """
    Generate a list of omicron trigger times that satisfy snr threshold

    Args:
        snr_thresh:
            SNR threshold above which glitches will be kept
        start:
            GPS time at which to begin looking for glitches
        stop:
            GPS time at which to stop looking for glitches
        window:
            Amount of time in seconds on either side of a glitch
            to query data for
        sample_rate:
            Sample rate of queried data, specified in Hz
        channel:
            Channel name used to read data. Should include the
            interferometer prefix
        trigger_files:
            List of h5 files containing omicron triggers
        chunk_size:
            Length in seconds of data to query at one time

    Returns:
        A list of glitch timeseries, a list of SNRs, and a list timestamps
    """

    authenticate()
    glitches = []
    snrs = []
    gpstimes = []

    with h5py.File(trigger_path) as f:
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

    logging.info(
        f"Querying {len(triggers)} triggers from {trigger_path}"
        f" over {stop - start} seconds of data "
    )

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
    triggers = triggers[mask]
    # query data for each trigger
    for trigger in triggers:
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

    if glitches:
        glitches = np.stack(glitches)
        with h5py.File(outfile, "w") as f:
            f.create_dataset("glitches", data=glitches)
            f.create_dataset("snrs", data=snrs)
            f.create_dataset("times", data=times)


def deploy_collect_glitches(
    trigger_paths: List[Path],
    pad: Tuple[float, float],
    snr_thresh: float,
    sample_rate: float,
    channel: str,
    ifo: str,
    outdir: Path,
    condordir: Path,
    accounting_group: str,
    accounting_group_user: str,
    request_memory: float = 32678,
):

    condordir = condordir / ifo
    condordir.mkdir(exist_ok=True, parents=True)
    # create text file from which the condor job will read
    # the trigger_path, and outfile for each job
    parameters = "start,stop,trigger_path,outfile\n"

    for f in trigger_paths:
        match = re_trigger_fname.search(f.name)
        if match is not None:
            t0, length = float(match.group("t0")), float(match.group("length"))
            t0, length = intify(t0), intify(length)
            logging.info(f"Generating glitch dataset for {ifo} and file {f}")
            start, stop = t0, t0 + length
            out = outdir / f"{ifo}-glitches-{t0}-{length}.hdf5"
            if out.exists():
                logging.info(f"Found existing glitch file {out}. Skipping")
                continue
            parameters += f"{start},{stop},{f},{out}\n"
        else:
            logging.warning(f"Could not parse trigger file name {f.name}")
            continue

    arguments = "--trigger-path $(trigger_path) --outfile $(outfile) "
    arguments += "--start $(start) --stop $(stop) "
    arguments += f"--pad {pad[0]} {pad[1]} --snr-thresh {snr_thresh} "
    arguments += f"--channel {channel} --sample-rate {sample_rate} "

    subfile = condor.make_submit_file(
        executable="collect-glitches",
        name="collect_glitches",
        parameters=parameters,
        arguments=arguments,
        submit_dir=condordir,
        accounting_group=accounting_group,
        accounting_group_user=accounting_group_user,
        clear=True,
        request_memory=f"ifthenelse(isUndefined(MemoryUsage), {request_memory}, int(3*MemoryUsage))",  # noqa
        periodic_release="(HoldReasonCode =?= 26 || HoldReasonCode =?= 34) && (JobStatus == 5)",  # noqa
        periodic_remove="(JobStatus == 1) && MemoryUsage >= 7G",
    )

    dag_id = condor.submit(subfile)
    logging.info(
        f"Launching collection of glitches for {ifo} with dag id {dag_id}"
    )
    condor.watch(dag_id, condordir, held=False)
    logging.info(f"Completed collection of glitches for {ifo} ")


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

    """
    Parses args into a format compatible for Pyomicron,
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
    accounting_group: str,
    accounting_group_user: str,
    analyze_testing_set: bool = False,
    force_generation: bool = False,
    verbose: bool = False,
):

    """
    Generate a set of glitches for both
    H1 and L1 that can be added to background.

    First, an omicron job is launched via pyomicron
    (https://github.com/gwpy/pyomicron/). Next, triggers (i.e. glitches)
    above a given SNR threshold are selected, and data is queried
    for these triggers and saved in an h5 file. This file contains
    a group for each interferometer in `ifos`. Within each group,
    the `times` dataset contains the GPS time of each glitch, the
    `snrs` dataset contains the SNR of each glitch, and the
    `glitches` dataset contains a timeseries of the strain data
    of length `2 * window * sample_rate`, with the time of the glitch
    centered within it.

    Args:
        snr_thresh:
            SNR threshold above which glitches will be kept
        start:
            GPS time at which to begin looking for glitches
        stop:
            GPS time at which to stop looking for glitches for the
            training dataset.
            Marks the beginning of the testing dataset
        test_stop:
            GPS time at which to stop looking for glitches for the
            testing dataset
        q_min:
            Minimum q value of tiles for omicron
        q_max:
            Maximum q value of tiles for omicron
        f_min:
            Lowest frequency for omicron to consider, specified in Hz
        cluster_dt:
            Time window for omicron to cluster neighboring triggers
        chunk_duration:
            Duration of data in seconds for PSD estimation
        segment_duration:
            Duration of data in seconds for FFT
        overlap:
            Overlap in seconds between neighbouring segments and chunks
        mismatch_max:
            Maximum distance between (Q, f) tiles
        window:
            Amount of time in seconds on either side of a glitch to
            query data for
        datadir:
            Directory to which the glitch dataset will be written
        logdir:
            Directory to which the log file will be written
        channel:
            Channel name used to read data. Should not include the
            interferometer prefix
        frame_type:
            Frame type for data discovery with gwdatafind
        sample_rate:
            Sample rate of queried data, specified in Hz
        state_flag:
            Identifier for which segments to use. Descriptions of flags
            and there usage can be found here:
            https://wiki.ligo.org/DetChar/DataQuality/AligoFlags
        ifos:
            List of interferometers to query data from. Expected to be given
            by prefix; e.g. "H1" for Hanford
        chunk_size:
            Length in seconds of data to query at one time
        analyze_testing_set:
            If True, get glitches for the testing dataset
        force_generation:
            If False, will not generate data if an existing dataset exists
        verbose:
            If True, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.

    Returns: The name of the file containing the glitch data
    """

    authenticate()

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

    train_futures = []

    run_dir = datadir / "condor" / "omicron"
    train_run_dir = run_dir / "train"
    test_run_dir = run_dir / "test"
    omicron_log_file = run_dir / "pyomicron.log"

    # seconds before and after trigger time such that we
    # we have exactly enough data to sample
    # the trigger time at the first or last sample of the kernel
    pad = (
        psd_length + kernel_length + (fduration / 2),
        kernel_length + (fduration / 2),
    )
    pool = ThreadPoolExecutor(4)

    for ifo in ifos:
        train_ifo_dir = train_run_dir / ifo
        test_ifo_dir = test_run_dir / ifo
        train_ifo_dir.mkdir(exist_ok=True, parents=True)

        # launch pyomicron futures for training set and testing sets.
        # as train futures complete, launch glitch generation processes.
        # let test set jobs run in background as glitch data is queried
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

    deploy_futures = []
    for future in as_completed(train_futures):
        ifo = future.result()
        outdir = datadir / "train" / "glitches" / ifo
        condordir = datadir / "condor" / "glitches"
        condordir.mkdir(exist_ok=True, parents=True)
        outdir.mkdir(exist_ok=True, parents=True)
        trigger_dir = train_run_dir / ifo / "merge" / f"{ifo}:{channel}"
        trigger_paths = sorted(list(trigger_dir.glob("*.h5")))

        args = [
            trigger_paths,
            pad,
            snr_thresh,
            sample_rate,
            f"{ifo}:{channel}",
            ifo,
            outdir,
            condordir,
            accounting_group,
            accounting_group_user,
        ]
        future = pool.submit(deploy_collect_glitches, *args)
        deploy_futures.append(future)

    for future in as_completed(deploy_futures):
        continue

    return datadir


if __name__ == "__main__":
    main()
