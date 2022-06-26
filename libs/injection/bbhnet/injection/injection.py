import logging
from pathlib import Path
from typing import List

import bilby
import numpy as np
import scipy.signal as sig
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from gwpy.timeseries import TimeSeries

from bbhnet.io import h5
from bbhnet.io.timeslides import TimeSlide


def calc_snr(data, noise_psd, fs, fmin=20):
    """Calculate the waveform SNR given the background noise PSD

    Args:
        data: timeseries of the signal whose SNR is to be calculated
        noise_psd: PSD of the background that the signal is in
        fs: sampling frequency of the signal and background
        fmin: minimum frequency for the highpass filter

    Returns:
        The SNR of the signal, a single value

    """

    data_fd = np.fft.rfft(data) / fs
    data_freq = np.fft.rfftfreq(len(data)) * fs
    dfreq = data_freq[1] - data_freq[0]

    noise_psd_interp = noise_psd.interpolate(dfreq)
    noise_psd_interp[noise_psd_interp == 0] = 1.0

    snr = 4 * np.abs(data_fd) ** 2 / noise_psd_interp.value * dfreq
    snr = np.sum(snr[fmin <= data_freq])
    snr = np.sqrt(snr)

    return snr


def _set_missing_params(default, supplied):
    common = set(default).intersection(supplied)
    res = {k: supplied[k] for k in common}
    for k in default.keys() - common:
        res.update({k: default[k]})
    return res


def get_waveform_generator(**waveform_generator_params):
    """Get a waveform generator using
    :meth:`bilby.gw.waveform_generator.WaveformGenerator`

    Args:
        waveform_generator_params: dict
        Keyword arguments to waveform generator
    """
    default_waveform_sampling_params = dict(
        duration=8,
        sampling_frequency=16384,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
    )
    default_waveform_approximant_params = dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=50,
        minimum_frequency=20,
    )

    sampling_params = _set_missing_params(
        default_waveform_sampling_params, waveform_generator_params
    )
    waveform_approximant_params = _set_missing_params(
        default_waveform_approximant_params, waveform_generator_params
    )

    sampling_params["waveform_arguments"] = waveform_approximant_params

    logging.debug("Waveform parameters: {}".format(sampling_params))
    return bilby.gw.waveform_generator.WaveformGenerator(**sampling_params)


def generate_gw(
    sample_params, waveform_generator=None, **waveform_generator_params
):
    """Generate raw gravitational-wave signals, pre-interferometer projection.

    Args:
        sample_params: dictionary of GW parameters
        waveform_generator: bilby.gw.WaveformGenerator with appropriate params
        waveform_generator_params: keyword arguments to
        :meth:`bilby.gw.WaveformGenerator`

    Returns:
        An (n_samples, 2, waveform_size) array, containing both polarizations
        for each of the desired number of samples. The first polarization is
        always plus and the second is always cross
    """

    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]
    n_samples = len(sample_params)

    waveform_generator = waveform_generator or get_waveform_generator(
        **waveform_generator_params
    )

    sample_rate = waveform_generator.sampling_frequency
    waveform_duration = waveform_generator.duration
    waveform_size = int(sample_rate * waveform_duration)

    num_pols = 2
    signals = np.zeros((n_samples, num_pols, waveform_size))

    filtered_signal = apply_high_pass_filter(
        signals, sample_params, waveform_generator
    )
    return filtered_signal


def apply_high_pass_filter(signals, sample_params, waveform_generator):
    sos = sig.butter(
        N=8,
        Wn=waveform_generator.waveform_arguments["minimum_frequency"],
        btype="highpass",
        output="sos",
        fs=waveform_generator.sampling_frequency,
    )
    polarization_names = None
    for i, p in enumerate(sample_params):
        polarizations = waveform_generator.time_domain_strain(p)
        if polarization_names is None:
            polarization_names = sorted(polarizations.keys())

        polarizations = np.stack(
            [polarizations[p] for p in polarization_names]
        )
        filtered = sig.sosfiltfilt(sos, polarizations, axis=1)
        signals[i] = filtered
    return signals


def project_raw_gw(
    raw_waveforms,
    sample_params,
    waveform_generator,
    ifo,
    get_snr=False,
    noise_psd=None,
):
    """Project a raw gravitational wave onto an intterferometer

    Args:
        raw_waveforms: the plus and cross polarizations of a list of GWs
        sample_params: dictionary of GW parameters
        waveform_generator: the waveform generator that made the raw GWs
        ifo: interferometer
        get_snr: return the SNR of each sample
        noise_psd: background noise PSD used to calculate SNR the sample

    Returns:
        An (n_samples, waveform_size) array containing the GW signals as they
        would appear in the given interferometer with the given set of sample
        parameters. If get_snr=True, also returns a list of the SNR associated
        with each signal
    """

    polarizations = {
        "plus": raw_waveforms[:, 0, :],
        "cross": raw_waveforms[:, 1, :],
    }

    sample_params = [
        dict(zip(sample_params, col)) for col in zip(*sample_params.values())
    ]
    n_sample = len(sample_params)

    sample_rate = waveform_generator.sampling_frequency
    waveform_duration = waveform_generator.duration
    waveform_size = int(sample_rate * waveform_duration)

    signals = np.zeros((n_sample, waveform_size))
    snr = np.zeros(n_sample)

    ifo = bilby.gw.detector.get_empty_interferometer(ifo)
    for i, p in enumerate(sample_params):

        # For less ugly function calls later on
        ra = p["ra"]
        dec = p["dec"]
        geocent_time = p["geocent_time"]
        psi = p["psi"]

        # Generate signal in IFO
        signal = np.zeros(waveform_size)
        for mode, polarization in polarizations.items():
            # Get ifo response
            response = ifo.antenna_response(ra, dec, geocent_time, psi, mode)
            signal += response * polarization[i]

        # Total shift = shift to trigger time + geometric shift
        dt = waveform_duration / 2.0
        dt += ifo.time_delay_from_geocenter(ra, dec, geocent_time)
        signal = np.roll(signal, int(np.round(dt * sample_rate)))

        # Calculate SNR
        if noise_psd is not None:
            if get_snr:
                snr[i] = calc_snr(signal, noise_psd, sample_rate)

        signals[i] = signal
    if get_snr:
        return signals, snr
    return signals


def inject_signals_into_timeslide(
    raw_timeslide: TimeSlide,
    out_timeslide: TimeSlide,
    ifos: List[str],
    prior_file: Path,
    spacing: float,
    waveform_duration: float,
    sample_rate: float,
    fmin: float,
    reference_frequency: float,
    waveform_approximant: float,
    snr_range: List[float],
    buffer: float = 0,
    fftlength: float = 2,
):

    """Injects simulated BBH signals into a TimeSlide object that represents
    timeshifted background data. Currently only supports h5 file format.

    Args:
        raw_timeslide: TimeSlide object of raw background data
        out_timeslide: TimeSlide object to store injections
        ifos: list of interferometers corresponding to timeseries
        prior_file: prior file for bilby to sample from
        spacing: seconds between each injection
        fmin: Minimum frequency for highpass filter
        waveform_duration: length of injected waveforms
        sample_rate: sampling rate
        snr_range: desired signal SNR range
        buffer:

    Returns:
        Paths to the injected files and the parameter file
    """

    # define a Bilby waveform generator

    # TODO: should sampling rate be automatically inferred
    # from raw data?
    waveform_generator = get_waveform_generator(
        waveform_approximant=waveform_approximant,
        reference_frequency=reference_frequency,
        minimum_frequency=fmin,
        sample_rate=sample_rate,
        duration=waveform_duration,
    )

    # initiate prior
    priors = bilby.gw.prior.BBHPriorDict(prior_file)

    for segment in raw_timeslide.segments:

        start = segment.t0
        stop = segment.tf

        # determine signal times
        # based on length of segment and spacing;
        # The signal time represents the first sample
        # in the signals generated by project_raw_gw.
        # not to be confused with the t0, which should
        # be the middle sample

        signal_times = np.arange(start + buffer, stop - buffer, spacing)
        n_samples = len(signal_times)

        # sample prior
        parameters = priors.sample(n_samples)

        # generate raw waveforms
        raw_signals = generate_gw(
            parameters, waveform_generator=waveform_generator
        )

        # dictionary to store
        # gwpy timeseries of background
        raw_ts = {}

        # dictionary to store injection
        # datasets to write to h5
        inj_datasets = {}

        # load segment;
        # expects that ifo is the name
        # of the dataset
        data = segment.load(datasets=ifos)

        # parse data based on ifos passed
        # into gwpy timeseries
        times = data[-1]

        for i, ifo in enumerate(ifos):
            raw_ts = TimeSeries(data[i], times=times)

            # calculate psd for this segment
            psd = raw_ts.psd(fftlength)

            # project raw waveforms
            signals, snr = project_raw_gw(
                raw_signals,
                parameters,
                waveform_generator,
                ifo,
                get_snr=True,
                noise_psd=psd,
            )

            # loop over signals, injecting them into the
            # raw strain

            for start, signal in zip(signal_times, signals):
                stop = start + len(signal) * sample_rate
                times = np.arange(start, stop, 1 / sample_rate)

                # create gwpy timeseries for signal
                signal = TimeSeries(signal, times=times)

                # inject into raw background
                raw_ts.inject(signal)

            inj_datasets[ifo] = raw_ts.value

        # now write this segment to out TimeSlide
        h5.write_timeseries(
            out_timeslide.root, prefix="inj", t=times, datasets=inj_datasets
        )

    return out_timeslide
