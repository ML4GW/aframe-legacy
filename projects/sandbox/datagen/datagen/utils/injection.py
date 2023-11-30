import logging
from typing import Dict, List, Tuple

import numpy as np
from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters,
    convert_to_lal_binary_neutron_star_parameters,
)
from bilby.gw.source import lal_binary_black_hole, lal_binary_neutron_star
from bilby.gw.waveform_generator import WaveformGenerator


def convert_to_detector_frame(samples):
    """Converts mass parameters from source to detector frame"""
    for key in ["mass_1", "mass_2", "chirp_mass", "total_mass"]:
        if key in samples:
            samples[key] = samples[key] * (1 + samples["redshift"])
    return samples


def generate_gw(
    sample_params: Dict[List, str],
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_duration: float,
    waveform_approximant: str,
    detector_frame_prior: bool = False,
):
    """Generate raw gravitational-wave signals, pre-interferometer projection.
    Args:
        sample_params:
            Dictionary of GW parameters where key is the parameter name
            and value is a list of the parameters
        minimum_frequency:
            Minimum frequency of the gravitational wave. The part
            of the gravitational wave at lower frequencies will
            not be generated. Specified in Hz.
        reference_frequency:
            Frequency of the gravitational wave at the state of
            the merger that other quantities are defined with
        sample_rate:
            Rate at which to sample time series, specified in Hz
        waveform_duration:
            Duration of waveform in seconds
        waveform_approximant:
            Name of waveform approximant to use.
        detector_frame_prior:
            If False, converts parameters to detector frame
    Returns:
        An (n_samples, 2, waveform_size) array, containing both polarizations
        for each of the desired number of samples.
        The waveforms are shifted such that
        the coalescence time lies at the center of the sample
    """

    if not detector_frame_prior:
        sample_params = convert_to_detector_frame(sample_params)

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


def generate_gw_bns(
    sample_params: Dict[List, str],
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_duration: float,
    waveform_approximant: str,
    detector_frame_prior: bool = False,
):
    # padding is used to take care of the wrap-around issue with the waveform
    # initally waveform is elongated by padding number of seconds
    # afterwads the padding length is chopped off to get the intended waveform
    padding = 1
    if not detector_frame_prior:
        sample_params = convert_to_detector_frame(sample_params)

    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]

    n_samples = len(sample_params)

    waveform_generator = WaveformGenerator(
        duration=waveform_duration + padding,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=lal_binary_neutron_star,
        parameter_conversion=convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments={
            "waveform_approximant": waveform_approximant,
            "reference_frequency": reference_frequency,
            "minimum_frequency": minimum_frequency,
        },
    )

    logging.info("Generating BNS waveforms     : {}".format(n_samples))

    waveform_size = int(sample_rate * waveform_duration)
    num_pols = 2
    signals = np.zeros((n_samples, num_pols, waveform_size))

    for i, p in enumerate(sample_params):
        polarizations = waveform_generator.time_domain_strain(p)
        polarization_names = sorted(polarizations.keys())
        polarizations = np.stack(
            [polarizations[p] for p in polarization_names]
        )

        # just shift the coalescence to the left by 200 datapoints
        # to cancel wraparound in the beginning
        dt = -200
        polarizations = np.roll(polarizations, dt, axis=-1)

        # cut off the first sec of the waveform where the wraparound occurs
        padding_length = padding * sample_rate
        signals[i] = polarizations[:, int(padding_length) :]

        # every 1000th waveform
        if not i % 1000:
            # note the following is  only called if verbose=True
            logging.debug(f"{i + 1} polarizations generated")
    logging.info("Finished generating polarizations")
    return signals


def inject_waveforms(
    background: Tuple[np.ndarray, np.ndarray],
    waveforms: np.ndarray,
    signal_times: np.ndarray,
) -> np.ndarray:

    """
    Inject a set of signals into background data

    Args:
        background:
            A tuple (t, data) of np.ndarray arrays.
            The first tuple is an array of times.
            The second tuple is the background strain values
            sampled at those times.
        waveforms:
            An np.ndarray of shape (n_waveforms, waveform_size)
            that contains the waveforms to inject
        signal_times:
            An array of times where signals will be injected. Corresponds to
            first sample of waveforms.
    Returns:
        A dictionary where the key is an interferometer and the value
        is a timeseries with the signals injected
    """

    times, data = background[0].copy(), background[1].copy()
    if len(times) != len(data):
        raise ValueError(
            "The times and background arrays must be the same length"
        )

    sample_rate = 1 / (times[1] - times[0])
    # create matrix of indices of waveform_size for each waveform
    num_waveforms, waveform_size = waveforms.shape
    idx = np.arange(waveform_size)[None] - int(waveform_size // 2)
    idx = np.repeat(idx, len(waveforms), axis=0)

    # offset the indices of each waveform corresponding to their time offset
    time_diffs = signal_times - times[0]
    idx_diffs = (time_diffs * sample_rate).astype("int64")
    idx += idx_diffs[:, None]

    # flatten these indices and the signals out to 1D
    # and then add them in-place all at once
    idx = idx.reshape(-1)
    waveforms = waveforms.reshape(-1)
    data[idx] += waveforms

    return data
