from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from bbhnet.io.timeslides import Segment

MAYBE_SEGMENTS = Union[Segment, Iterable[Segment]]


# TODO: move these functions to library?
def load_segments(segments: MAYBE_SEGMENTS, dataset: str):
    """
    Quick utility function which just wraps a Segment's
    `load` method so that we can execute it in a process
    pool since methods aren't picklable.
    """
    # allow iterable of segments
    # for case where we want to load
    # background and injection together

    if isinstance(segments, Segment):
        segments.load(dataset)

    else:
        for segment in segments:
            segment.load(dataset)
    return segments


def get_write_dir(
    write_dir: Path,
    shift: Union[str, Segment],
    label: str,
    norm: Optional[float] = None,
) -> Path:
    """
    Quick utility function for getting the name of the directory
    to which to save the outputs from an analysis using a particular
    time-shift/norm-seconds combination
    """

    if isinstance(shift, Segment):
        shift = shift.shift

    if norm is not None:
        write_dir = write_dir / shift / f"{label}-norm-seconds.{norm}"
    else:
        write_dir = write_dir / shift / f"{label}"
    write_dir.mkdir(parents=True, exist_ok=True)
    return write_dir


def get_fname(t: np.ndarray, write_dir: Path, prefix: str = "out"):
    """Infer the file name produced by write_timeseries
    from a time array
    """
    t0 = t[0]
    t0 = int(t0) if int(t0) == t0 else t0

    length = t[-1] - t[0] + t[1] - t[0]
    length = int(length) if int(length) == length else length

    # format the filename and write the data to an archive
    fname = write_dir / f"{prefix}_{t0}-{length}.hdf5"
    return fname
