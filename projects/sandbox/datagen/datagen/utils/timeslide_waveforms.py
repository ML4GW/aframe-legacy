import logging
import time
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch

from bbhnet.analysis.ledger.injections import (
    InjectionParameterSet,
    LigoResponseSet,
)
from ml4gw.spectral import normalize_psd


def calc_shifts_required(
    segments: List[Tuple[int, int]], Tb: float, shift: float
):
    """
    Based off of the lengths of the segments and the
    amount of data that will need to be sloughed off
    the ends due to shifting, calculate how many shifts
    will be required to achieve Tb seconds worth of background

    Args:
        segments: A list of tuples of the start and stop times of the segments
        Tb: The amount of background data to generate
        shift: The increment to shift the data by

    Returns the number of shifts required to achieve Tb seconds of background
    """

    livetime = sum([stop - start for start, stop in segments])
    n_segments = len(segments)
    shifts_required = 0
    while True:
        max_shift = shift * shifts_required
        total_livetime = (livetime - n_segments * max_shift) * shifts_required
        if total_livetime < Tb:
            shifts_required += 1
            continue
        break

    return shifts_required


def io_with_blocking(f, fname, timeout=10):
    start_time = time.time()
    while True:
        try:
            return f(fname)
        except BlockingIOError:
            if (time.time() - start_time) > timeout:
                raise


def get_waveform_contents(fname):
    with h5py.File(fname, "r") as f:
        metadata = ["length", "num_injections", "sample_rate", "duration"]
        return tuple([f.attrs[i] for i in metadata])


def get_rejected_contents(fname):
    with h5py.File(fname, "r") as f:
        return f.attrs["length"]


def merge_output(results_dir: Path, fname: Path, dtype=np.float64):
    # go through our files once up front to infer
    # a few scalar attributes we'll need to
    # initialize the full dataset
    num_waveforms, num_injections, num_rejected = 0, 0, 0
    for f in results_dir.glob("tmp-*.h5"):
        if "rejected" in f.name:
            num_rejected += io_with_blocking(get_rejected_contents, f)
            continue

        waveforms, injections, sample_rate, duration = io_with_blocking(
            get_waveform_contents, f
        )
        num_waveforms += waveforms
        num_injections += injections
    waveform_size = int(sample_rate * duration)

    # initialize the output file with a bunch of
    # empty datasets that we'll write to directly
    fields = LigoResponseSet.__dataclass_fields__
    rejected_fname = fname.parent / "rejected-params.h5"
    target = h5py.File(fname, "w")
    rejected = h5py.File(rejected_fname, "w")
    with target, rejected:
        # start with some of the attributes
        target.attrs["num_injections"] = num_injections
        target.attrs["length"] = num_waveforms
        target.attrs["duration"] = duration
        target.attrs["sample_rate"] = sample_rate
        waveforms = target.create_group("waveforms")
        parameters = target.create_group("parameters")

        rejected.create_group("parameters")
        rejected.attrs["length"] = num_rejected

        logging.info(
            f"Initializing HDF5 structure with {num_waveforms} waveforms"
        )
        for key, attr in fields.items():
            shape = (num_waveforms,)

            if attr.metadata["kind"] == "metadata":
                continue
            elif attr.metadata["kind"] == "waveform":
                shape += (waveform_size,)
                group = waveforms
            else:
                group = parameters
                if key == "shift":
                    shape += (2,)

                if key in InjectionParameterSet.__dataclass_fields__:
                    rejected["parameters"].create_dataset(
                        key, shape=(num_rejected,), dtype=dtype
                    )
            group.create_dataset(key, shape=shape, dtype=dtype)
        logging.info("HDF5 datasets initialized")

        idx, ridx = 0, 0
        for source in results_dir.glob("tmp-*.h5"):
            with h5py.File(source, "r") as src:
                length = src.attrs["length"]
                if "rejected" in source.name:
                    slc = np.s_[ridx : ridx + length]
                    for key in src["parameters"]:
                        x = src["parameters"][key][:]
                        rejected["parameters"][key].write_direct(
                            x, dest_sel=slc
                        )
                    ridx += length
                    # source.unlink()
                    continue

                slc = np.s_[idx : idx + length]
                for key, attr in fields.items():
                    if attr.metadata["kind"] == "metadata":
                        continue
                    else:
                        group = attr.metadata["kind"] + "s"
                        x = src[group][key][:]
                        target[group][key].write_direct(x, dest_sel=slc)
            idx += length
            # source.unlink()


# def merge_output(results_dir: Path, fname: Path):
#     files = results_dir.glob("tmp-*.h5")
#     response_set = LigoResponseSet()
#     for f in files:
#         fset = io_with_blocking(LigoResponseSet.read, f)
#         response_set.append(fset)
#         f.unlink()
#     response_set.write(fname)


def load_psds(
    background: Path, ifos: List[str], sample_rate: float, df: float
):
    with h5py.File(background, "r") as f:
        psds = []
        for ifo in ifos:
            hoft = f[ifo][:]
            psd = normalize_psd(hoft, df, sample_rate)
            psds.append(psd)
    psds = torch.tensor(np.stack(psds), dtype=torch.float64)
    return psds


def calc_segment_injection_times(
    start: float,
    stop: float,
    spacing: float,
    buffer: float,
    waveform_duration: float,
):
    """
    Calculate the times at which to inject signals into a segment

    Args:
        start: The start time of the segment
        stop: The stop time of the segment
        spacing: The spacing between signals
        jitter: The jitter to apply to the signal times
        buffer: The buffer to apply to the start and end of the segment
        waveform_duration: The duration of the waveform

    Returns np.ndarray of injection times
    """

    buffer += waveform_duration // 2
    spacing += waveform_duration
    injection_times = np.arange(start + buffer, stop - buffer, spacing)
    return injection_times
