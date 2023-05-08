import time
from pathlib import Path
from typing import List

from lal import gpstime
from mldatafind.io import fetch_timeseries
from mldatafind.segments import query_segments
from type.utils import make_dummy
from typeo import scriptify

from bbhnet.architectures import architectures
from bbhnet.trainer import train


def find_segments(
    channels: List[str],
    segment_names: List[str],
    start: float,
    cadence: float,
    duration: float,
    wait: int = 120,  # length of time dqsegdb takes to update
):
    """
    Iterator that returns training segments
    """
    segments = []
    while True:
        stop = int(gpstime.gps_time_now())
        segments = query_segments(
            segment_names,
            start,
            stop,
            duration,
        )
        if segments:
            # if multiple segments, use most recent segment;
            # TODO: should we enforce segment duration is
            # equal to requested train_duration?
            start, stop = segments[-1]

            data = fetch_timeseries(channels, start, stop, array_like=True)
            yield start, stop, data

            start += cadence

        time.sleep(wait)


def get_glitches(start: float, stop: float):
    pass


def get_waveforms(path: Path):
    pass


def export(weights: Path):
    pass


# arguments to exclude
exclude = []


# TODO: add new typeo as submodule
@scriptify(
    kwargs=make_dummy(train, exclude=exclude),
    architecture=architectures,
)
def main(
    segment_names: List[str],
    channels: List[str],
    train_duration: float,
    cadence: float,
    waveform_dataset: Path,
    **kwargs
):

    start = int(gpstime.gps_time_now())
    segment_it = find_segments(
        channels,
        segment_names,
        start,
        cadence,
        train_duration,
    )

    for start, stop, background_data in segment_it:
        # logic for organizing glitch dataset for training
        # based on start, stop of training background data

        glitches = get_glitches(start, stop)

        # since this will presumably be a fixed dataset,
        # should we just pass the path to the file to train?
        waveforms = get_waveforms(waveform_dataset)

        weights = train(background_data, glitches, waveforms, **kwargs)
        export(weights)
