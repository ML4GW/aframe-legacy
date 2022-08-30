from typing import Dict, List

import numpy as np
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator
from gwpy.timeseries import TimeSeries


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


def inject_waveforms(
    background_data: Dict[str, np.ndarray],
    times: np.ndarray,
    waveforms: np.ndarray,
    signal_times: np.ndarray,
    sample_rate: float,
) -> Dict[str, np.ndarray]:

    """
    Inject a set of signals into background data

    Args:
        background_data:
            A dictionary where the key is an interferometer
            and the value is a timeseries
        times:
            times corresponding to samples in background_data
        waveforms:
            A dictionary where the key is an interfereometer
            and the value is an np.ndarray array of
            projected waveforms of shape (n_signals, waveform_size)
        signal_times: np.ndarray,:
            An array of times where signals will be injected
        sample_rate:
            timeseries sampling rate
    Returns
        A dictionary where the key is an interferometer and the value
        is a timeseries with the signals injected
    """
    output = {}

    for i, (ifo, x) in enumerate(background_data.items()):
        signals = waveforms[:, i, :]
        ts = TimeSeries(x, times=times)

        # loop over signals, injecting them into the raw strain
        for signal_start, signal in zip(signal_times, signals):
            signal_stop = signal_start + len(signal) * (1 / sample_rate)
            signal_times = np.arange(
                signal_start, signal_stop, 1 / sample_rate
            )

            # create gwpy timeseries for signal
            signal = TimeSeries(signal, times=signal_times)

            # inject into raw background
            ts.inject(signal)
        output[ifo] = ts.value
    return output
