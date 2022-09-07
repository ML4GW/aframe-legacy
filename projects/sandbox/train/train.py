from math import pi
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from bbhnet.data.dataloader import BBHInMemoryDataset, GlitchSampler
from bbhnet.data.transforms import HighpassFilter, WhiteningTransform
from bbhnet.distributions import Cosine, LogNormal, Uniform
from bbhnet.logging import configure_logging
from bbhnet.trainer import trainify
from ml4gw.transforms import RandomWaveformInjection


def split(X, frac, axis):
    return np.split(X, [int(frac * X.shape[axis])], axis=axis)


# note that this function decorator acts both to
# wrap this function such that the outputs of it
# (i.e. the training and possible validation data)
# get passed as inputs to deepclean.trainer.trainer.train,
# as well as to expose these arguments _as well_ as those
# from bbhnet.trainer.trainer.train to command line
# execution and parsing
@trainify
def main(
    glitch_dataset: str,
    signal_dataset: str,
    hanford_background: str,
    livingston_background: str,
    waveform_frac: float,
    glitch_frac: float,
    kernel_length: float,
    highpass: float,
    sample_rate: float,
    batch_size: int,
    device: str,
    outdir: Path,
    logdir: Path,
    mean_snr: float = 8,
    std_snr: float = 4,
    min_snr: Optional[float] = 2,
    batches_per_epoch: Optional[int] = None,
    fduration: Optional[float] = None,
    trigger_distance: float = 0,
    valid_frac: Optional[float] = None,
    valid_stride: Optional[float] = None,
    verbose: bool = False,
    **kwargs
):
    """
    waveform_frac:
        The fraction of waveforms in each batch
    glitch_frac:
        The fraction of glitches in each batch
    sample_rate:
        The rate at which all relevant input data has
        been sampled
    kernel_length:
        The length, in seconds, of each batch element
        to produce during iteration.
    min_snr:
        Minimum SNR value for sampled waveforms.
    max_snr:
        Maximum SNR value for sampled waveforms.
    highpass:
        Frequencies above which to keep
    batch_size:
        Number of samples to produce during at each
        iteration
    batches_per_epoch:
        The number of batches to produce before raising
        a `StopIteration` while iteratingkernel_length:
    fduration:
        duration of the time domain filter used
        to whiten the data (If using WhiteningTransform).
        Note that fduration / 2 seconds will be cropped from
        both ends of kernel_length
    trigger_distance_size:

    """

    # make out dir and configure logging file
    outdir.mkdir(exist_ok=True, parents=True)
    logdir.mkdir(exist_ok=True, parents=True)

    configure_logging(logdir / "train.log", verbose)

    # TODO: maybe package up hanford and livingston
    # (or any arbitrary set of ifos) background files into one
    # for simplicity

    if valid_frac is not None:
        frac = 1 - valid_frac
    else:
        frac = None

    # initiate training glitch sampler
    with h5py.File(glitch_dataset, "r") as f:
        h1_glitches = glitch_dataset["H1_glitches"][:]
        l1_glitches = glitch_dataset["L1_glitches"][:]
    glitch_inserter = GlitchSampler(
        prob=glitch_frac,
        max_offset=int(trigger_distance * sample_rate),
        H1=h1_glitches,
        L1=l1_glitches,
    )

    # initiate training waveform sampler
    with h5py.File(signal_dataset, "r") as f:
        signals = f["signals"][:]
        if frac is not None:
            raise ValueError
            # signals, valid_signals = split(signals, frac, 0)
            # valid_plus, valid_cross = valid_signals.transpose(1, 0, 2)

            # ra = f["ra"][-len(valid_signals):]
            # dec = f["dec"][-len(valid_signals):]
            # psi = f["psi"][-len(valid_signals):]
            # snr = f["snr"][-len(valid_signals):]

        plus, cross = signals.transpose(1, 0, 2)

    injector = RandomWaveformInjection(
        dec=Cosine(),
        psi=Uniform(0, pi),
        phi=Uniform(-pi, pi),
        snr=LogNormal(mean_snr, std_snr, min_snr),
        sample_rate=sample_rate,
        highpass=highpass,
        prob=waveform_frac,
        trigger_offset=trigger_distance,
        plus=plus,
        cross=cross,
    )
    augmentation = torch.nn.Sequential(glitch_inserter, injector)

    background = []
    for fname in [hanford_background, livingston_background]:
        with h5py.File(fname, "r") as f:
            hoft = f["hoft"][:]
        background.append(hoft)
    background = np.stack(background)

    if frac is not None:
        background, valid_background = split(background, frac, 1)
    injector.fit(H1=background[0], L1=background[1])

    # create full training dataloader
    train_dataset = BBHInMemoryDataset(
        background,
        int(kernel_length * sample_rate),
        batch_size=batch_size,
        stride=1,
        batches_per_epoch=batches_per_epoch,
        preprocessor=augmentation,
        coincident=False,
        shuffle=True,
        device=device,
    )

    # TODO: hard-coding num_ifos into preprocessor. Should
    # we just expose this as an arg? How will this fit in
    # to the broader-generalization scheme?
    whitener = WhiteningTransform(
        2, sample_rate, kernel_length, highpass=highpass, fduration=fduration
    )
    whitener.fit(background)
    hpf = HighpassFilter(highpass, sample_rate)
    preprocessor = torch.nn.Sequential([whitener, hpf])

    # deterministic validation glitch sampler
    if valid_frac is not None:
        raise ValueError
    else:
        valid_dataset = None

    return train_dataset, valid_dataset, preprocessor
