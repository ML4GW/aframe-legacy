from pathlib import Path
from typing import List

import h5py
import lal
from bbhnet_omicron.condor import get_executable, make_submit_file, submit
from mldatafind.io import filter_and_sort_files
from typeo import scriptify


@scriptify
def update_glitch_dataset(
    datadir: Path,
    archive_dir: Path,
    ifos: List[str],
    look_back: float,
    max_glitches: int,
):

    # begin by adding the most recent glitches since the last update
    now = lal.gpstime.gpstime_now()
    for ifo in ifos:
        with h5py.File(datadir / "glitches.h5") as f:
            gpstimes = f[ifo]["gpstime"][:]

        latest = max(gpstimes)
        latest_day = latest // 100000

        subdirs = list(archive_dir.iterdir())
        mask = subdirs >= latest_day
        subdirs = subdirs[mask]

        all_files = []
        for dir in subdirs:
            files = dir.glob("*.h5")
            files = filter_and_sort_files(files, latest, now)
            all_files.append(files)

        all_files = sorted(all_files)

        # TODO: count number of new triggers.
        # keep most recent max_glitches - num_new_triggers
        # of the last glitch dataset and combine with new triggers


@scriptify
def deploy(
    # glitch dataset args
    data_dir: Path,
    archive_dir: Path,
    ifos: List[str],
    look_back: float,
    max_glitches: int,
    # deployment args
    submit_dir: Path,
    runevery: int,
    offset: int,
    preptime: int,
    logdir: Path,
    accounting_group: str,
    accounting_user: str,
    universe: str,
):
    executable = get_executable("update-glitches")
    arguments = [
        "--data-dir",
        str(data_dir),
        "--archive-dir",
        str(archive_dir),
        "--ifos",
        " ".join(ifos),
        "--look-back",
        str(look_back),
        "--max-glitches",
        str(max_glitches),
    ]
    arguments = " ".join(arguments)
    kwargs = {"request_memory": "1024", "request_disk": "1024"}
    name = "update-glitches"
    subfile = make_submit_file(
        arguments,
        runevery,
        offset,
        preptime,
        logdir,
        accounting_group,
        accounting_user,
        universe,
        executable,
        name,
        submit_dir,
        **kwargs
    )

    submit(subfile)
