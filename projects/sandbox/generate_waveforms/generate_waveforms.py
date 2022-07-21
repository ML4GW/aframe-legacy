import logging
from pathlib import Path
from typing import Dict, List, Optional

import bilby
import h5py
import numpy as np

from bbhnet.injection import generate_gw
from bbhnet.injection.utils import get_waveform_generator
from bbhnet.logging import configure_logging
from hermes.typeo import typeo


def write_waveforms(
    fname: Path,
    waveforms: np.ndarray,
    params: Dict[str, List[float]],
    sample_rate: float,
):
    with h5py.File(fname, "w") as f:
        # write signals attributes, snr, and signal parameters
        for param, values in params.items():
            f.create_dataset(param, data=values)

        f.create_dataset("signals", data=values)
        f.attrs.update({"sample_rate": sample_rate})


@typeo
def main(
    prior_file: str,
    n_samples: int,
    outdir: Path,
    data_dir: Path,
    waveform_duration: float = 8,
    sample_rate: float = 4096,
    valid_frac: Optional[float] = None,
    force_generation: bool = False,
    verbose: bool = False,
) -> List[Path]:
    """Simulates a set of raw BBH signals and saves them to an output file.

    Args:
        prior_file: prior file for bilby to sample from
        n_samples: number of signal to inject
        outdir: output directory to which signals will be written
        waveform_duration: length of injected waveforms
        sample_rate: sample rate of the signal in Hz
        force_generation: if True, generate signals even if path already exists
    Returns:
        path to output file
    """

    # make output dir
    outdir.mkdir(exist_ok=True, parents=True)
    configure_logging(outdir / "generate_waveforms.log", verbose)

    # check if we want to use any of our waveforms for validation
    signal_files = [data_dir / "train_signals.h5"]
    if valid_frac is not None:
        if not 0 < valid_frac < 1:
            raise ValueError(
                f"'valid_frac' must be between 0 and 1, not {valid_frac}"
            )
        signal_files.append(data_dir / "valid_signals.h5")

    # if all the necessary files already exist and
    # we're not forcing ourselves to generate more,
    # go ahead and just return the filenames
    for fname in signal_files:
        if not fname.exists() or force_generation:
            break
    else:
        logging.info(
            "Signal files {} already exist".format(", ".join(signal_files))
        )
        return signal_files

    # log and print out some simulation parameters
    logging.info("Simulation parameters")
    logging.info("Number of samples     : {}".format(n_samples))
    logging.info("Sample rate [Hz]      : {}".format(sample_rate))
    logging.info("Prior file            : {}".format(prior_file))

    # define a Bilby waveform generator
    waveform_generator = get_waveform_generator(
        duration=waveform_duration, sampling_frequency=sample_rate
    )

    # sample GW parameters from prior distribution
    priors = bilby.gw.prior.PriorDict(prior_file)
    sample_params = priors.sample(n_samples)
    signals = generate_gw(sample_params, waveform_generator=waveform_generator)

    # Write params and similar to output file
    if np.isnan(signals).any():
        raise ValueError("The signals contain NaN values")

    if valid_frac is not None:
        # split off some of our waveforms for validation
        num_valid = int(valid_frac * n_samples)
        signals, valid_signals = np.split(signals, [-num_valid])

        # break apart the corresponding params as well
        train_params, valid_params = {}, {}
        for param, values in sample_params.items():
            train_params[param] = values[:-num_valid]
            valid_params[param] = values[-num_valid:]

        # reset sample_params to the train_params we kept
        sample_params = train_params

        write_waveforms(
            signal_files[1], valid_signals, valid_params, sample_rate
        )

    # write the train signals regardless
    write_waveforms(signal_files[0], signals, sample_params, sample_rate)
    return signal_files


if __name__ == "__main__":
    main()
