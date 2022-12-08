import logging
from pathlib import Path
from typing import Iterable

from gwpy.timeseries import TimeSeriesDict
from mldatafind import query_segments
from mldatafind.authenticate import authenticate
from typeo import scriptify

from bbhnet.logging import configure_logging


def intify(x: float):
    return int(x) if int(x) == x else x


@scriptify
def main(
    start: float,
    stop: float,
    sample_rate: float,
    channels: Iterable[str],
    state_flags: Iterable[str],
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

    # create credentials to access LIGO data products
    authenticate()

    logdir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "generate_background.log", verbose)

    prefix = "background"
    n_matches = len(list(datadir.glob(f"{prefix}*.h5")))

    if n_matches > 1:
        raise ValueError(
            "f{n_matches} background files found. Only 1 should exists"
        )

    if n_matches == 1 and not force_generation:
        logging.info(
            "Background data already exists"
            " and forced generation is off. Not generating background"
        )
        return

    segment_start, segment_stop = query_segments(
        state_flags, start, stop, minimum_length
    )[0]

    # TODO: utility function in mldatafind
    # for infering file name from TimeSeriesDict
    duration = intify(segment_stop - segment_start)
    start = intify(segment_start)

    path = datadir / f"{prefix}-{start}-{duration}.h5"

    ts_dict = TimeSeriesDict.get(channels, segment_start, segment_stop)
    ts_dict.resample(sample_rate)
    ts_dict.write(path)

    return path
