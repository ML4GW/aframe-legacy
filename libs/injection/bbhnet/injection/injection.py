from typing import Dict, List

import numpy as np
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator


def generate_gw(
    sample_params: Dict[List, str],
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_duration: float,
    waveform_approximant: str,
):
    """Generate raw gravitational-wave signals, pre-interferometer projection.
    Args:
        sample_params: dictionary of GW parameters
        minimum_frequency:
            minimum_frequency for generating waveform; not to be confused with
            highpass filter frequency
        reference_frequency: reference frequency for generating waveform
        sample_rate: rate at which to sample time series
        waveform_duration: duration of waveform
        waveform_approximant: name of waveform approximant to use.
    Returns:
        An (n_samples, 2, waveform_size) array, containing both polarizations
        for each of the desired number of samples.
        The waveforms are shifted such that
        the coalescence time lies at the center of the sample
    """

    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]

    n_samples = len(sample_params)

    waveform_generator = WaveformGenerator(
        duration=waveform_duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": waveform_approximant,
            "reference_frequency": reference_frequency,
            "minimum_frequency": minimum_frequency,
        },
    )

    waveform_size = int(sample_rate * waveform_duration)

    num_pols = 2
    signals = np.zeros((n_samples, num_pols, waveform_size))

    for i, p in enumerate(sample_params):
        polarizations = waveform_generator.time_domain_strain(p)
        polarization_names = sorted(polarizations.keys())
        polarizations = np.stack(
            [polarizations[p] for p in polarization_names]
        )

        # center so that coalescence time is middle sample
        dt = waveform_duration / 2
        polarizations = np.roll(polarizations, int(dt * sample_rate), axis=-1)
        signals[i] = polarizations

    return signals
