from collections import OrderedDict
from math import pi
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from bbhnet.data.dataloader import BBHInMemoryDataset
from bbhnet.data.distributions import Cosine, LogNormal, Uniform
from bbhnet.data.glitch_sampler import GlitchSampler
from bbhnet.data.transforms import HighpassFilter, WhiteningTransform
from bbhnet.logging import configure_logging
from bbhnet.trainer import trainify
from ml4gw.transforms import RandomWaveformInjection


class MultiInputSequential(torch.nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def split(X, frac, axis):
    return np.split(X, [int(frac * X.shape[axis])], axis=axis)


def prepare_augmentation(
    glitch_dataset: Path,
    waveform_dataset: Path,
    glitch_prob: float,
    waveform_prob: float,
    sample_rate: float,
    highpass: float,
    mean_snr: float,
    std_snr: float,
    min_snr: Optional[float] = None,
    trigger_distance: float = 0,
    valid_frac: Optional[float] = None,
):
    # build a glitch sampler from a pre-saved bank of
    # glitches which will randomly insert them into
    # either or both interferometer channels
    with h5py.File(glitch_dataset, "r") as f:
        h1_glitches = f["H1_glitches"][:]
        l1_glitches = f["L1_glitches"][:]
    glitch_inserter = GlitchSampler(
        prob=glitch_prob,
        max_offset=int(trigger_distance * sample_rate),
        H1=h1_glitches,
        L1=l1_glitches,
    )

    # initiate a waveform sampler from a pre-saved bank
    # of GW waveform polarizations which will randomly
    # project them to inteferometer responses and
    # inject those resposnes into the input data
    with h5py.File(waveform_dataset, "r") as f:
        signals = f["signals"][:]

        # TODO: right now signals are prepared such that the
        # coalescence time is at the end of the window, so roll
        # them to put them in the middle as expected. Do we want
        # to do this here or in the generate_waveforms project?
        signals = np.roll(signals, -signals.shape[-1] // 2, axis=-1)
        if valid_frac is not None:
            raise ValueError
            # signals, valid_signals = split(signals, 1 - valid_frac, 0)
            # valid_plus, valid_cross = valid_signals.transpose(1, 0, 2)

            # ra = f["ra"][-len(valid_signals):]
            # dec = f["dec"][-len(valid_signals):]
            # psi = f["psi"][-len(valid_signals):]
            # snr = f["snr"][-len(valid_signals):]

        plus, cross = signals.transpose(1, 0, 2)

    # instantiate source parameters as callable
    # distributions which will produce samples
    injector = RandomWaveformInjection(
        dec=Cosine(),
        psi=Uniform(0, pi),
        phi=Uniform(-pi, pi),
        snr=LogNormal(mean_snr, std_snr, min_snr),
        sample_rate=sample_rate,
        highpass=highpass,
        prob=waveform_prob,
        trigger_offset=trigger_distance,
        plus=plus,
        cross=cross,
    )

    # stack glitch inserter and waveform sampler into
    # a single random augmentation object which will
    # be called at data-loading time (i.e. won't be
    # used on validation data).
    # TODO: return validation data if valid_frac not None
    return MultiInputSequential(
        OrderedDict(
            [("glitch_inserter", glitch_inserter), ("injector", injector)]
        )
    )


def prepare_preprocessor(
    num_ifos: int,
    sample_rate: float,
    kernel_length: float,
    fduration: float,
    highpass: float,
):
    whitener = WhiteningTransform(
        num_ifos,
        sample_rate,
        kernel_length,
        highpass=highpass,
        fduration=fduration,
    )

    hpf = HighpassFilter(highpass, sample_rate)
    return torch.nn.Sequential(
        OrderedDict([("whitener", whitener), ("higpass", hpf)])
    )


def load_background(*backgrounds: Path):
    # TODO: maybe package up hanford and livingston
    # (or any arbitrary set of ifos) background files into one
    # for simplicity
    background = []
    for fname in backgrounds:
        with h5py.File(fname, "r") as f:
            hoft = f["hoft"][:]
        background.append(hoft)
    return np.stack(background)


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
    min_snr: Optional[float] = None,
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

    # build a torch module that we'll use for doing
    # random augmentation at data-loading time
    augmenter = prepare_augmentation(
        glitch_dataset,
        signal_dataset,
        glitch_prob=glitch_frac,
        waveform_prob=waveform_frac,
        sample_rate=sample_rate,
        highpass=highpass,
        mean_snr=mean_snr,
        std_snr=std_snr,
        min_snr=min_snr,
        trigger_distance=trigger_distance,
        valid_frac=valid_frac,
    )

    # TODO: maybe package up hanford and livingston
    # (or any arbitrary set of ifos) background files into one
    # for simplicity
    background = load_background(hanford_background, livingston_background)
    if valid_frac is not None:
        background, valid_background = split(background, 1 - valid_frac, 1)

    # fit our waveform injector to this background
    # to facilitate the SNR remapping
    augmenter._modules["injector"].fit(H1=background[0], L1=background[1])
    for module in augmenter._modules.values():
        module.to(device)

    # create full training dataloader
    train_dataset = BBHInMemoryDataset(
        background,
        int(kernel_length * sample_rate),
        batch_size=batch_size,
        stride=1,
        batches_per_epoch=batches_per_epoch,
        preprocessor=augmenter,
        coincident=False,
        shuffle=True,
        device=device,
    )

    # TODO: hard-coding num_ifos into preprocessor. Should
    # we just expose this as an arg? How will this fit in
    # to the broader-generalization scheme?
    preprocessor = prepare_preprocessor(
        num_ifos=2,
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        highpass=highpass,
        fduration=fduration,
    )

    # fit the whitening module to the background then
    # move eveyrthing to the desired device
    preprocessor._modules["whitener"].fit(background)
    for module in preprocessor._modules.values():
        module.to(device)

    # deterministic validation glitch sampler
    if valid_frac is not None:
        raise ValueError
    else:
        valid_dataset = None

    return train_dataset, valid_dataset, preprocessor
