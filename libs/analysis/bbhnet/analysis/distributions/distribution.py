from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Union

import h5py
import numpy as np

if TYPE_CHECKING:
    from bbhnet.io.timeslides import Segment, TimeSlide


@dataclass
class Distribution:
    dataset: str

    def __post_init__(self):
        self.Tb = 0

    def read(self, fname: Union[str, Path]):
        with h5py.File(fname, "r") as f:
            x = f[self.dataset][:].reshape(-1)
            t = f["GPSstart"][:]
            return x, t

    def update(self, x: np.ndarray, t: np.ndarray):
        """Update this distribution to reflect new data"""

        raise NotImplementedError

    def nb(self, threshold: float):
        """Number of events in this distribution above a threshold"""

        raise NotImplementedError

    def far(self, threshold: float, analysis_time: float) -> float:
        """see https://arxiv.org/pdf/1508.02357.pdf, eq. 17"""

        nb = self.nb(threshold)
        time_ratio = analysis_time / self.Tb
        return 1 - np.exp(-time_ratio * (1 + nb))

    def characterize_events(
        self,
        segment: "Segment",
        dataset: str,
        event_times: Union[float, Iterable[float]],
        window_length: float = 1,
        kernel_length: float = 1,
    ):
        # duck-typing check on whether there are
        # multiple events in the segment or just the one.
        # Even if there's just one but it's passed as an
        # iterable, we'll record return a 2D array, otherwise
        # just return 1D
        try:
            event_iter = iter(event_times)
            single_event = False
        except TypeError:
            event_iter = iter([event_times])
            single_event = True

        y, t = segment.load(dataset)
        t = t + kernel_length
        sample_rate = 1 / (t[1] - t[0])
        window_size = int(window_length * sample_rate)

        fars, times = [], []
        for event_time in event_iter:
            # start with the first timestep that could
            # have contained the event in the NN input
            idx = ((t - event_time) > 0).argmax()
            event_far = self.far(y[idx : idx + window_size], segment.length)

            fars.append(event_far)
            times.append(t[idx : idx + window_size] - event_time)

        fars, times = np.stack(fars), np.stack(times)
        if single_event:
            return fars[0], times[0]
        return fars, times

    def fit(
        self,
        timeslides: Union["TimeSlide", Iterable["TimeSlide"]],
        warm_start: bool = True,
    ) -> None:
        if not warm_start:
            self.__post_init__()

        # TODO: accept pathlike and initialize a timeslide?
        if isinstance(timeslides, TimeSlide):
            timeslides = [timeslides]

        for slide in timeslides:
            for segment in slide:
                for fname in segment:
                    y, t = self.read(fname)
                    self.update(y, t)
