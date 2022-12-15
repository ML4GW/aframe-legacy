import logging
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
from datagen.utils.injection import generate_gw
from typeo import scriptify

from bbhnet.logging import configure_logging


@scriptify
def main(
    n_samples: int,
    logdir: Path,
    datadir: Path,
    reference_frequency: float,
    minimum_frequency: float,
    sample_rate: float,
    waveform_duration: float,
    prior: Optional[Callable] = None,
    parameter_file: Optional[Path] = None,
    waveform_approximant: str = "IMRPhenomPv2",
    force_generation: bool = False,
    verbose: bool = False,
):
    """Simulates a set of raw BBH signals and saves them to an output file.

    Args:
        prior_file: prior file for bilby to sample from
        n_samples: number of signal to inject
        logdir: directory where log file will be written
        datadir: output directory to which signals will be written
        reference_frequency: reference frequency for waveform generation
        minimum_frequency: minimum_frequency for waveform generation
        sample_rate: rate at which to sample waveforms
        waveform_duration: length of injected waveforms
        waveform_approximant: which lalsimulation waveform approximant to use
        force_generation: if True, generate signals even if path already exists
        verbose: log verbosely
    """

    # make dirs
    datadir.mkdir(exist_ok=True, parents=True)
    logdir.mkdir(exist_ok=True, parents=True)

    configure_logging(logdir / "generate_waveforms.log", verbose)

    # check if signal file already exists
    signal_file = datadir / "signals.h5"

    if signal_file.exists() and not force_generation:
        logging.info(
            "Signal data already exists and forced generation is off. "
            "Not generating signals."
        )
        return signal_file

    # log and print out some simulation parameters
    logging.info("Simulation parameters")
    logging.info("Number of samples     : {}".format(n_samples))
    logging.info("Sample rate [Hz]      : {}".format(sample_rate))

    # sample gw parameters from prior distribution
    if prior is not None:
        logging.info(
            "Prior name            : {}".format(prior.__class__.__name__)
        )
        priors = prior()
        sample_params = priors.sample(n_samples)

    # sample gw parameters from a catalog of simulated events
    # these parameters will override any that were also
    # specified by the prior class
    if parameter_file is not None:
        logging.info("Parameter file        : {}".format(parameter_file))
        with h5py.File(parameter_file, "r") as f:
            events = f["events"]
            num_events = len(events)
            event_idx = np.sort(
                np.random.choice(num_events, n_samples, replace=False)
            )
            field_names = events.dtype.names

            for i, name in enumerate(field_names):
                sample_params[name] = events[event_idx][name]

    signals = generate_gw(
        sample_params,
        minimum_frequency,
        reference_frequency,
        sample_rate,
        waveform_duration,
        waveform_approximant,
    )

    # Write params and similar to output file
    if np.isnan(signals).any():
        raise ValueError("The signals contain NaN values")

    with h5py.File(signal_file, "w") as f:
        # write signals attributes, snr, and signal parameters
        for k, v in sample_params.items():
            f.create_dataset(k, data=v)

        f.create_dataset("signals", data=signals)

        # write attributes
        f.attrs.update(
            {
                "size": n_samples,
                "sample_rate": sample_rate,
                "waveform_duration": waveform_duration,
                "waveform_approximant": waveform_approximant,
                "reference_frequency:": reference_frequency,
                "minimum_frequency": minimum_frequency,
            }
        )

    return signal_file


if __name__ == "__main__":
    main()
