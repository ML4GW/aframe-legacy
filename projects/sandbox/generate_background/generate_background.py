import logging
from pathlib import Path
from typing import List

import numpy as np
from gwdatafind import find_urls
from gwpy.segments import DataQualityDict
from gwpy.timeseries import TimeSeries

from bbhnet.io.h5 import write_timeseries
from bbhnet.logging import configure_logging
from hermes.typeo import typeo


@typeo
def main(
    start: float,
    stop: float,
    ifos: List[str],
    sample_rate: float,
    channel: str,
    frame_type: str,
    state_flag: str,
    minimum_length: float,
    datadir: Path,
    logdir: Path,
    force_generation: bool = False,
    verbose: bool = False,
):
    """Generates a stretch of background data for training BBHnet,
    as well as a stretch of backround for generating timeslides and
    injections for testing

    Uses segments defined by `state_flag` to query
    a continuous and coincident stretch of data between
    `start` and `stop` that is at least 2 * `minimum_length`
    seconds long. The first half of this queried segment
    will be used for training BBHnet while the second half
    will be used for creating timeslides and injections
    used for testing.

    Args:
        start: start gpstime
        stop: stop gpstime
        ifos: List of interferometers
        sample_rate: sample rate
        channel: data channel
        frame_type: frame type for gwdatafind
        state_flag: name of science segments to use
        minimum_length:
            minimum continuous, coincident stretch
            of time between start and stop
        datadir: directory to store data
        logdir: directory to store logs
        force_generation:
            If False, will only generate data if output path doesnt exist.
            If True, will always generate data
        verbose: If True, log verbosely
    """

    logdir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)

    configure_logging(logdir / "generate_background.log", verbose)

    # if force generation is False check to see
    # if both training and testing backgrounds
    # exist
    train_file_exists = len(list(datadir.glob("training_background*.h5"))) > 0
    test_file_exists = len(list(datadir.glob("testing_background*.h5"))) > 0

    if train_file_exists and test_file_exists and not force_generation:
        logging.info(
            "All background data already exists"
            " and forced generation is off. Not generating background"
        )
        return

    # query segments for each ifo
    # I think a certificate is needed for this
    segments = DataQualityDict.query_dqsegdb(
        [f"{ifo}:{state_flag}" for ifo in ifos],
        start,
        stop,
    )

    # create copy of first ifo segment list to start
    intersection = segments[f"{ifos[0]}:{state_flag}"].active.copy()

    # loop over ifos finding segment intersection
    for ifo in ifos:
        intersection &= segments[f"{ifo}:{state_flag}"].active

    # find first continuous segment of 2 * minimum length
    # the first minimum_length of this segment will be
    # used for training BBHnet, the second will be
    # used for creating timeslides and injections for testing
    segment_lengths = np.array(
        [float(seg[1] - seg[0]) for seg in intersection]
    )

    continuous_segments = np.where(segment_lengths >= minimum_length)[0]

    if len(continuous_segments) == 0:
        raise ValueError(
            "No segments of minimum length, not producing background"
        )

    # choose first of such segments
    segment = intersection[continuous_segments[0]]

    logging.info(
        "Querying coincident, continuous segment "
        "from {} to {}".format(*segment)
    )

    seg_start, seg_stop = segment
    midpoint = (seg_start + seg_stop) / 2

    training_data = {}
    testing_data = {}

    for ifo in ifos:

        # find frame files
        files = find_urls(
            site=ifo.strip("1"),
            frametype=f"{ifo}_{frame_type}",
            gpsstart=start,
            gpsend=stop,
            urltype="file",
        )
        ts = TimeSeries.read(
            files, channel=f"{ifo}:{channel}", start=seg_start, end=seg_stop
        )

        # resample
        ts = ts.resample(sample_rate)

        if np.isnan(ts).any():
            raise ValueError(
                f"The background for ifo {ifo} contains NaN values"
            )

        # store first half for training
        train_ts = ts.crop(seg_start, midpoint)
        training_data[ifo] = train_ts

        # store second half for testing
        test_ts = ts.crop(midpoint, seg_stop)
        testing_data[ifo] = test_ts

        train_times = train_ts.times.value
        test_times = test_ts.times.value

    train_fname = write_timeseries(
        datadir, "training_background", t=train_times, **training_data
    )

    test_fname = write_timeseries(
        datadir, "testing_background", t=test_times, **testing_data
    )

    return train_fname, test_fname
