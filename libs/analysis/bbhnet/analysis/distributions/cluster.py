import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import h5py
import numpy as np

from bbhnet.analysis.distributions.distribution import Distribution


@dataclass
class ClusterDistribution(Distribution):
    """
    Distribution representing a clustering of sampled
    points over consecutive windows of length `t_clust`
    Args:
        t_clust: The length of the clustering window
    """

    t_clust: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events = []
        self.event_times = []
        self.shifts = []

    def _load(self, path: Path):
        """Load distribution information from an HDF5 file"""
        with h5py.File(path, "r") as f:
            self.events = f["events"][:]
            self.event_times = f["event_times"][:]
            self.shifts = f["shifts"][:]
            self.fnames = list(f["fnames"][:])
            self.Tb = f["Tb"]
            t_clust = f.attrs["t_clust"]
            if t_clust != self.t_clust:
                raise ValueError(
                    "t_clust of Distribution object {t_clust}"
                    "does not match t_clust of data in h5 file {self.t_clust}"
                )

    def write(self, path: Path):
        """Write the distribution's data to an HDF5 file"""
        with h5py.File(path, "w") as f:
            f["events"] = self.events
            f["event_times"] = self.event_times
            f["shifts"] = self.shifts
            f["fnames"] = list(map(str, self.fnames))
            f["Tb"] = self.Tb
            f.attrs.update({"t_clust": self.t_clust})

    @classmethod
    def from_file(cls, dataset: str, path: Path):
        """Create a new distribution with data loaded from an HDF5 file"""
        with h5py.File(path, "r") as f:
            t_clust = f.attrs["t_clust"]
        obj = cls(dataset, t_clust)
        obj._load(path)
        return obj

    def update(self, x: np.ndarray, t: np.ndarray, shift: float):
        """
        Update the histogram using the values from `x`,
        and update the background time `Tb` using the
        values from `t`. It is assumed that `t` is contiguous
        in time.
        """

        # update livetime before we start
        # manipulating the t array
        self.Tb += t[-1] - t[0] + t[1] - t[0]

        # cluster values over window length
        sample_rate = 1 / (t[1] - t[0])

        # samples per cluster window
        clust_size = int(sample_rate * self.t_clust)

        # take care of reshaping
        # into windows of cluster size
        extra = len(x) % clust_size
        if extra != 0:
            arg_max = np.argmax(x[-extra:])
            time = t[-extra:][arg_max]
            max_ = x[-extra:][arg_max]
            x = x[:-extra]
            t = t[:-extra]
            self.events.append(max_)
            self.event_times.append(time)
            self.shifts.append(shift)

        x = x.reshape((-1, clust_size))
        maxs = list(np.amax(x, axis=-1))
        arg_maxs = np.argmax(x, axis=-1)
        times = np.take_along_axis(
            t.reshape((-1, clust_size)),
            np.expand_dims(arg_maxs, axis=-1),
            axis=-1,
        ).squeeze(axis=-1)

        # update events, event times,
        # livetime and timeshifts
        self.events.extend(maxs)
        self.event_times.extend(times)
        self.shifts.extend(np.zeros_like(times) + shift)

    def nb(self, threshold: Union[float, np.ndarray]):
        """
        Counts the number of events above the indicated
        `threshold`
        """
        events = np.array(self.events)
        if isinstance(threshold, np.ndarray):
            nb = [np.sum(events >= thresh) for thresh in threshold]
        else:
            nb = np.sum(events >= threshold)

        logging.debug(
            "Threshold {} has {} events greater than it "
            "in distribution {}".format(threshold, nb, self)
        )
        return np.array(nb)
