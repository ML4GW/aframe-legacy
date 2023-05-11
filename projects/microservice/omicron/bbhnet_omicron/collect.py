import logging
from pathlib import Path
from typing import List

import h5py
import lal
from bbhnet_omicron.condor import get_executable, make_submit_file, submit
from mldatafind.find import find_data
from mldatafind.io import filter_and_sort_files
from typeo import scriptify


@scriptify
def update_glitch_dataset(
    datadir: Path,
    archive_dir: Path,
    ifos: List[str],
    channel: str,
    chunk_size: float,
    sample_rate: float,
    window: float,
    snr_thresh: float,
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

        times = []
        for dir in subdirs:
            files = dir.glob("*.h5")
            files = filter_and_sort_files(files, latest, now)
            for file in files:
                with h5py.File(file) as f:
                    triggers = f["triggers"][:]
                    times.append(triggers["time"][:])

        # fetch timeseries we will crop to create glitches
        start, stop = min(times) - window, max(times) + window
        generator = next(
            find_data(
                [(start, stop)],
                [f"{ifo}:{channel}"],
                chunk_size=chunk_size,
            )
        )

        glitches = []
        snrs = []
        gpstimes = []
        for data in generator:
            # restrict to triggers within current data chunk
            data = data.resample(sample_rate)[channel]
            times = data.times.value
            mask = triggers["time"] > times[0] + window
            mask &= triggers["time"] < times[-1] - window

            chunk_triggers = triggers[mask]
            # query data for each trigger
            for trigger in chunk_triggers:
                time = trigger["time"]
                try:
                    glitch_ts = data.crop(time - window, time + window)
                except ValueError:
                    logging.warning(
                        f"Data not available for trigger at time: {time}"
                    )
                    continue
                else:
                    glitches.append(list(glitch_ts.value))
                    snrs.append(trigger["snr"])
                    gpstimes.append(time)

        n_new_glitches = len(glitches)
        remaining = max_glitches - n_new_glitches
        print(remaining, n_new_glitches)
        # TODO: count number of new triggers,
        # keep most recent max_glitches - num_new_triggers
        # from the last glitch dataset and combine with new triggers


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
        **kwargs,
    )

    submit(subfile)
