import logging
from pathlib import Path
from typing import List

import h5py
from mldatafind.authenticate import authenticate
from mldatafind.io import fetch_timeseries
from mldatafind.segments import query_segments
from typeo import scriptify

from aframe.logging import configure_logging


def _intify(x: float):
    return int(x) if int(x) == x else x


def _make_fname(prefix, t0, length):
    t0 = _intify(t0)
    length = _intify(length)
    return f"{prefix}-{t0}-{length}.hdf5"


def validate_train_file(
    filename: Path,
    ifos: List[str],
    sample_rate: float,
    train_start: float,
    train_stop: float,
    minimum_train_length: float,
):
    # if there exists files in training range,
    # check the timestamp and verify that it
    # meets the requested conditions
    with h5py.File(filename, "r") as f:
        missing_keys = [i for i in ifos if i not in f]
        if missing_keys:
            raise ValueError(
                "Background file {} missing data from {}".format(
                    filename, ", ".join(missing_keys)
                )
            )

        x = f[ifos[0]]
        t0 = x.attrs["x0"][()]
        length = len(x) / sample_rate

    in_range = train_start <= t0 <= (train_stop - minimum_train_length)
    long_enough = length >= minimum_train_length
    if not (in_range and long_enough):
        raise ValueError(
            "Background file {} has t0 {} and length {}s, "
            "which isn't compatible with request of {}s "
            "segment between {} and {}".format(
                filename,
                t0,
                length,
                minimum_train_length,
                train_start,
                train_stop,
            )
        )


@scriptify
def main(
    train_start: float,
    train_stop: float,
    test_stop: float,
    minimum_train_length: float,
    minimum_test_length: float,
    ifos: List[str],
    sample_rate: float,
    channel: str,
    state_flag: str,
    datadir: Path,
    logdir: Path,
    force_generation: bool = False,
    verbose: bool = False,
):
    """Generates background data for training and testing aframe

    Args:
        train_start: GPS time of the beginning of the training period
        train_stop: GPS time of the end of the training period.
            Also corresponds to the beginning of the testing period
        test_stop: GPS time of the end of the testing period
        minimum_train_length: The shortest a contiguous segment of training
            background can be. Specified in seconds
        minimum_test_length: The shortest a contiguous segment of testing
            background can be. Specified in seconds
        ifos: List of interferometers to query data from. Expected to be given
            by prefix; e.g. "H1" for Hanford
        sample_rate: Sample rate to which the timesires will be resampled
        channel: Channel from which to fetch the timeseries
        state_flag: Identifier for which segments to use
        datadir: Directory to which data will be written
        logdir: Directory to which the log file will be written
        force_generation: If false, will not generate data if an existing
            dataset exists
        verbose:
            If true, logs verbosely

    Returns: The name of the data directory
    """
    # make logdir dir
    logdir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "generate_background.log", verbose)

    # first query segments that meet minimum length
    # requirement during the requested training period
    authenticate()
    train_segments = query_segments(
        [f"{ifo}:{state_flag}" for ifo in ifos],
        train_start,
        train_stop,
        minimum_train_length,
    )
    if not train_segments:
        raise ValueError(
            "No segments of minimum length, not producing background"
        )

    test_segments = query_segments(
        [f"{ifo}:{state_flag}" for ifo in ifos],
        train_stop,
        test_stop,
        minimum_test_length,
    )

    segments = train_segments + test_segments
    channels = [f"{ifo}:{channel}" for ifo in ifos]

    for start, stop in segments:
        # using start/stops to decide if something
        # is a training file or not to make robust
        # to future use of multiple training background
        is_train = train_start <= start <= (train_stop - minimum_train_length)
        if is_train:
            subdir = "train"
            stop = min(stop, train_stop)
        else:
            subdir = "test"

        write_dir = datadir / subdir / "background"
        write_dir.mkdir(parents=True, exist_ok=True)
        fname = _make_fname("background", start, stop - start)
        write_path = write_dir / fname

        if write_path.exists() and not force_generation:
            if is_train:
                validate_train_file(
                    write_path,
                    ifos,
                    sample_rate,
                    train_start,
                    train_stop,
                    minimum_train_length,
                )

            logging.info(
                "Skipping download of segment {}-{}, already "
                "cached in file {}".format(start, stop, fname)
            )
            continue

        logging.info(
            "Downloading segment {}-{} to file {}".format(
                start, stop, write_path
            )
        )
        data = fetch_timeseries(channels, start, stop)
        data = data.resample(sample_rate)
        for ifo in ifos:
            data[ifo] = data.pop(f"{ifo}:{channel}")

        logging.info("Segment downloaded, writing to file")
        data.write(write_path)

        logging.info(f"Finished caching {write_path}")
    return datadir
