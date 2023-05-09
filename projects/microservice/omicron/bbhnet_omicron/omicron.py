import configparser
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from bbhnet_omicron.condor import get_executable, make_submit_file, submit
from mldatafind.authenticate import authenticate
from omicron.cli.process import main as omicron_main
from typeo import scriptify

# logic for deployment largely lifted from
# https://git.ligo.org/detchar/ligo-omicron/


@scriptify
def omicron_online(
    run_dir: Path,
    ifo: str,
    accounting_group: str,
    log_file: Path,
    config_file: Path,
    verbose: bool,
    group: str = "GW",
):

    """Parses args into a format compatible for Pyomicron,
    then launches omicron dag
    """

    # set run directory based on current time
    month = datetime.now().strftime("%Y-%m")
    day = datetime.now().strftime("%Y%m%d-%H%M%S")

    segfile_old = run_dir / "segments.txt"
    run_dir = run_dir / ifo / month / day
    run_dir.mkdir(exist_ok=True, parents=True)
    segfile_new = run_dir / "segments.txt"

    # if segments.txt exists, copy it to the new run directory
    # so that pyomicron will pick it up and start where it left off
    if segfile_old.exists():
        shutil.copy(segfile_old, segfile_new)

    # parse args into command line format
    # expected by omicron main
    omicron_args = [
        group,
        "--log-file",
        str(log_file),
        "--config-file",
        str(config_file),
        "--reattach",
        "--ifo",
        ifo,
        "-c",
        "request_disk=100M",
        "--output-dir",
        str(run_dir),
        "--archive",
        "--skip-rm",
        "--condor-accounting-group",
        accounting_group,
    ]
    if verbose:
        omicron_args += ["--verbose", "--verbose"]

    omicron_main(omicron_args)

    # keep track of where pyomicron left off
    # for next iteration of the run
    if segfile_new.exists():
        shutil.copy(segfile_new, segfile_old)


def make_config_file(
    run_dir: Path,
    ifo: str,
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
):
    # create config file
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
    config.set(section, "state-flag", f"{ifo}:{state_flag}")

    config_file_path = run_dir / f"omicron_{ifo}.ini"

    # write config file
    with open(config_file_path, "w") as configfile:
        config.write(configfile)
    return config_file_path


@scriptify
def deploy(
    # omicron options
    run_dir: Path,
    ifos: List[str],
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
    log_file: Path,
    archivedir: Path,
    # deployment options
    runevery: int,
    accounting_group: str,
    accounting_user: str,
    universe: str,
    # misc options
    verbose: bool,
):
    # authenticate, set the omicron_home environment
    # variable to point to directory where trigger
    # archive will be made, and make sure run_directory exists
    authenticate()
    archivedir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=True, parents=True)
    executable = get_executable("omicron-online")

    # work out random timing
    offset = random.randint(0, runevery - 1)
    preptime = offset * 60

    for ifo in ifos:
        ifo_dir = run_dir / ifo
        ifo_dir.mkdir(exist_ok=True, parents=True)
        # make omicron config file for this ifo
        config_file = make_config_file(
            ifo_dir,
            ifo,
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
        )
        fname = ifo_dir / f"omicron_driver_{ifo}.sub"
        arguments = [
            "--run-dir",
            str(run_dir),
            "--ifo",
            str(ifo),
            "--accounting-group",
            str(accounting_group),
            "--log-file",
            str(log_file),
            "--config-file",
            str(config_file),
            "--verbose",
        ]
        arguments = " ".join(arguments)
        logdir = ifo_dir / "logs"
        logdir.mkdir(exist_ok=True, parents=True)
        subfile = make_submit_file(
            arguments,
            runevery,
            offset,
            preptime,
            "GW",
            logdir,
            archivedir,
            accounting_group,
            accounting_user,
            universe,
            executable,
            fname,
        )

        submit(subfile)
