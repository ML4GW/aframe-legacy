from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np

from bbhnet.io.timeslides import Segment

SECONDS_IN_YEAR = 31556952

SEGMENT_LIKE = Union[Segment, Iterable[Segment], Tuple[np.ndarray, np.ndarray]]


@dataclass
class Distribution:
    dataset: str

    def __post_init__(self):
        self.Tb = 0
        self.events = []
        self.event_times = []
        self.shifts = []

    def write(self, path: Path):
        raise NotImplementedError

    def update(self, x: np.ndarray, t: np.ndarray):
        """Update this distribution to reflect new data"""

        raise NotImplementedError

    def nb(self, threshold: float):
        """Number of events in this distribution above a threshold"""

        raise NotImplementedError

    def far(self, threshold: float) -> float:
        """Compute the false alarm rate in units of yrs^{-1}"""

        nb = self.nb(threshold)
        return SECONDS_IN_YEAR * nb / self.Tb

    def significance(self, threshold: float, T: float) -> float:
        """see https://arxiv.org/pdf/1508.02357.pdf, eq. 17

        Represents the likelihood that at least one event
        with detection statistic value `threshold` will occur
        after observing this distribution for a period `T`.

        Args:
            threshold: The detection statistic to compare against
            T:
                The length of the analysis period in which the
                detection statistic was measured, in seconds
        """

        nb = self.nb(threshold)
        return 1 - np.exp(-T * (1 + nb) / self.Tb)

    def fit(
        self,
        segments: SEGMENT_LIKE,
        warm_start: bool = True,
    ) -> None:
        """
        Fit the distribution to the data contained in
        one or more `Segments`.

        Args:
            segments:
                `Segment` or list of `Segments` on which
                to update the distribution
            warm_start:
                Whether to fit the distribution from scratch
                or continue from its existing state.
        """
        if not warm_start:
            self.__post_init__()

        if isinstance(segments, Tuple):
            self.update(*segments)
            return

        if isinstance(segments, Segment):
            segments = [segments]

        for segment in segments:
            y, t = segment.load(self.dataset)
            self.update(y, t)

    def apply_vetoes(self, **vetoes: np.ndarray):
        """Remove events if the time in any of the interferometers lies in
        a vetoed segment.

        Args:
            vetoes:
                np.ndarray of shape (n_segments, 2)
        """

    def __str__(self):
        return f"{self.__class__.__name__}('{self.dataset}', Tb={self.Tb})"
