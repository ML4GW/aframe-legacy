import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import h5py
import numpy as np
import pycondor
import torch
from datagen.utils.injection import generate_gw
from mldatafind.segments import query_segments
from typeo import scriptify

from ml4gw.gw import (
    compute_network_snr,
    compute_observed_strain,
    get_ifo_geometry,
)
from ml4gw.spectral import normalize_psd


def load_psds(*backgrounds: Path, sample_rate: float, df: float):

    psds = []
    for fname in backgrounds:
        with h5py.File(fname, "r") as f:
            hoft = f["hoft"][:]
            psd = normalize_psd(hoft, df, sample_rate)
            psds.append(psd)
    return torch.stack(psds)


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

    buffer += waveform_duration // 2 + buffer
    spacing = waveform_duration + spacing
    injection_times = np.arange(start + buffer, stop - buffer, spacing)
    return injection_times


@scriptify
def main(
    start: float,
    stop: float,
    hanford_background: Path,
    livingston_background: Path,
    spacing: float,
    buffer: float,
    waveform_duration: float,
    cosmology: Callable,
    prior: Callable,
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_approximant: str,
    highpass: float,
    snr_threshold: float,
    ifos: List[str],
    output_fname: Path,
):
    """
    Generates the waveforms for a single segment
    """

    cosmology = cosmology()
    prior, detector_frame_prior = prior(cosmology)

    injection_times = calc_segment_injection_times(
        start, stop, spacing, buffer, waveform_duration
    )
    n_samples = len(injection_times)

    signals = []
    parameters = defaultdict(list)

    tensors, vertices = get_ifo_geometry(*ifos)
    df = 1 / waveform_duration
    psds = load_psds(
        hanford_background,
        livingston_background,
        sample_rate,
        df,
    )

    # loop until we've generated enough signals
    # with large enough snr to fill the segment,
    # keeping track of the number of signals rejected
    n_rejected = 0
    while len(signals) < n_samples:

        params = prior.sample(n_samples)
        params["geocent_time"] = injection_times
        waveforms = generate_gw(
            parameters,
            minimum_frequency,
            reference_frequency,
            sample_rate,
            waveform_duration,
            waveform_approximant,
            detector_frame_prior,
        )
        polarizations = {
            "cross": torch.Tensor(waveforms[:, 0, :]),
            "plus": torch.Tensor(waveforms[:, 1, :]),
        }
        projected = compute_observed_strain(
            torch.Tensor(params["dec"]),
            torch.Tensor(params["psi"]),
            torch.Tensor(params["ra"]),
            tensors,
            vertices,
            sample_rate,
            **polarizations,
        )
        snrs = compute_network_snr(projected, psds, sample_rate, highpass)
        snrs = snrs.numpy()
        mask = snrs > snr_threshold
        projected = projected[mask]
        n_rejected += np.sum(~mask)
        signals.append(waveforms)

        for key, value in params.items():
            parameters[key].extend(list(value[mask]))

    waveforms = torch.cat(waveforms)
    waveforms = waveforms[:n_samples]
    for key, value in params.items():
        parameters[key] = value[:n_samples]

    with h5py.File(output_fname, "a") as f:
        for k, v in parameters.items():
            f.create_dataset(k, data=v)

        f.attrs.update(
            {
                "n_rejected": n_rejected,
            }
        )
    return output_fname


def calc_shifts_required(
    segments: List[Tuple[int, int]], Tb: float, shift: float
):
    """
    Based off of the lengths of the segments and the
    amount of data that will need to be sloughed off
    the ends due to shifting, calculate how many shifts
    will be required to achieve Tb seconds worth of background
    """

    livetime = np.sum([stop - start for start, stop in segments])
    n_segments = len(segments)
    shifts_required = 1
    while True:
        max_shift = shift * shifts_required
        total_livetime = (livetime - n_segments * max_shift) * shifts_required
        if total_livetime < Tb:
            shifts_required += 1
            continue
        break

    return shifts_required


# until typeo update gets in just take all the same parameter as main
@scriptify
def deploy(
    start: float,
    stop: float,
    state_flag: str,
    Tb: float,
    shifts: Iterable[float],
    spacing: float,
    buffer: float,
    min_segment_length: float,
    cosmology: str,
    waveform_duration: float,
    prior: str,
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_approximant: str,
    highpass: float,
    snr_threshold: float,
    ifos: List[str],
    datadir: Path,
    logdir: Path,
    accounting_group_user: str,
    accounting_group: str,
):

    logdir.mkdir(exist_ok=True, parents=True)
    datadir = datadir / "timeslide_waveforms"

    hanford_background = datadir / "H1_background.h5"
    livingston_background = datadir / "L1_background.h5"

    # where condor info and sub files will live
    condor_dir = datadir / "condor"
    condor_log_dir = str(condor_dir / "logs")

    # condor_log_dir.mkdir(exist_ok=True, parents=True)
    condor_dir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)

    # query segments and calculate shifts required
    # to accumulate desired background livetime

    state_flags = [f"{ifo}:{state_flag}" for ifo in ifos]
    segments = query_segments(state_flags, start, stop, min_segment_length)
    shifts_required = calc_shifts_required(segments, Tb, max(shifts))

    # TODO: does this logic generalize to negative shifts?
    max_shift = max(shifts) * shifts_required

    # create text file from which the condor job will read
    # the start, stop, and shift for each job
    with open(condor_dir / "segments.txt", "w") as f:
        for start, stop in segments:
            for i in range(shifts_required):
                f.write(f"{start},{stop - max_shift}\n")

    executable = shutil.which("generate-timeslide-waveforms")

    # TODO: have typeo do this argument construction?
    arguments = "--start $(start) --stop $(stop)"
    arguments += f"--hanford-background {hanford_background}"
    arguments += f"--livingston-background {livingston_background}"
    arguments += f"--spacing {spacing} --buffer {buffer}"
    arguments += f"--waveform-duration {waveform_duration}"
    arguments += f"--minimum-frequency {minimum_frequency}"
    arguments += f"--reference-frequency {reference_frequency}"
    arguments += f"--sample-rate {sample_rate}"
    arguments += f"--waveform-approximant {waveform_approximant}"
    arguments += (
        f"--highpass {highpass} --snr_threshold {snr_threshold} --ifos {ifos}"
    )
    arguments += f"--prior {prior} --cosmology {cosmology}"
    arguments += f"--output_fname {datadir}/$(ProcID).hdf5"

    extra_lines = ["queue start,stop from segments.txt"]
    extra_lines.append(f"accounting_group_user = {accounting_group_user}")
    extra_lines.append(f"accounting_group = {accounting_group}")

    # create pycondor job
    job = pycondor.Job(
        name="timeslide_waveforms",
        executable=executable,
        universe="vanilla",
        error=condor_log_dir,
        output=condor_log_dir,
        log=condor_log_dir,
        submit=str(condor_dir),
        # TODO: can probably do some intelligent memory / disk requests
        request_memory=4 * 1024,
        request_disk=1024,
        getenv=True,
        arguments=arguments,
        extra_lines=extra_lines,
    )

    job.build_submit()
