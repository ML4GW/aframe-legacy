import logging
from pathlib import Path
from typing import List

from mldatafind.find import find_data
from typeo import scriptify

from bbhnet.logging import configure_logging


@scriptify
def main(
    start: float,
    stop: float,
    sample_rate: float,
    channels: List[str],
    state_flags: List[str],
    minimum_length: float,
    datadir: Path,
    logdir: Path,
    force_generation: bool = False,
    verbose: bool = False,
):
    """Generate background strain for training BBHnet

    Finds the first contiguous, coincident segment
    consistent with `segment_names`, and `minimum_length`,
    and queries strain data from `channels`

    Args:
        start:
            Start gpstime
        stop:
            Stop gpstime
        sample_rate:
            Rate to sample strain data
        channels:
            Strain channels to query
        state_flags:
            Name of segments
        minimum_length:
            Minimum segment length
        datadir:
            Directory to store data
        logdir:
            Directory to store log file
        force_generation:
            Force data to be generated even if path exists
        verbose:
            Log verbosely

    Returns:
        Path to data
    """

    logdir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "generate_background.log", verbose)

    path = datadir / "background.h5"

    if path.exists() and not force_generation:
        logging.info(
            "Background data already exists"
            " and forced generation is off. Not generating background"
        )
        return

    # create generator that will query data that satisfies
    # segment criteria and retrieve data from the first segment
    data_iterator = find_data(
        start,
        stop,
        channels,
        minimum_length,
        state_flags,
        retain_order=True,
        thread=True,
    )

    ts_dict = next(data_iterator)
    ts_dict.resample(sample_rate)
    ts_dict.write(path)

    return path
