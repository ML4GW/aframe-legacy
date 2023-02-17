from collections import defaultdict
from pathlib import Path
from typing import Callable, List

import h5py
import numpy as np
import torch
from typeo import scriptify

from bbhnet.injection import generate_gw
from ml4gw.gw import (
    compute_network_snr,
    compute_observed_strain,
    get_ifo_geometry,
)


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
    spacing: float,
    buffer: float,
    waveform_duration: float,
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

    prior, detector_frame_prior = prior()
    injection_times = calc_segment_injection_times(
        start, stop, spacing, buffer, waveform_duration
    )
    n_samples = len(injection_times)

    signals = []
    parameters = defaultdict(list)

    tensors, vertices = get_ifo_geometry(*ifos)
    psds = "tmp"

    # loop until we've generated enough signals
    # with large enough snr to fill the segment,
    # keeping track of the number of signals rejected
    n_rejected = 0
    while len(signals) < n_samples:

        params = prior.sample(n_samples)
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
            **polarizations
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
