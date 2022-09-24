import logging
from collections import OrderedDict
from math import ceil, pi
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from bbhnet.architectures import Preprocessor
from bbhnet.data.dataloader import BBHInMemoryDataset
from bbhnet.data.distributions import Cosine, LogNormal, Uniform
from bbhnet.data.glitch_sampler import GlitchSampler
from bbhnet.logging import configure_logging
from bbhnet.trainer import trainify
from ml4gw.transforms import RandomWaveformInjection


class MultiInputSequential(torch.nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def split(X, frac, axis):
    if isinstance(X, np.ndarray):
        fn = np.split
        splits = [int(frac * X.shape[axis])]
        kwargs = {"axis": axis}
    else:
        fn = torch.split
        kwargs = {"dim": axis}
        front = int(frac * X.shape[axis])
        splits = [front, X.shape[axis] - front]
    return fn(X, splits, **kwargs)


def make_validation_dataset(
    background,
    glitch_sampler,
    waveform_sampler,
    kernel_length: float,
    stride: float,
    sample_rate: float,
    glitch_frac: float,
):
    kernel_size = int(kernel_length * sample_rate)
    stride_size = int(stride * sample_rate)
    num_kernels = (background.shape[-1] - kernel_size) // stride_size + 1
    num_kernels = int(num_kernels)
    unfold_op = torch.nn.Unfold((1, num_kernels), dilation=(1, stride_size))

    num_ifos = len(background)
    background = background[:, : num_kernels * stride_size + kernel_size]
    background = torch.Tensor(background)[None, :, None]
    background = unfold_op(background)
    background = background.reshape(num_kernels, num_ifos, kernel_size)

    # construct a tensor of background with glitches inserted
    # overlap glitch_frac fraction of them
    h1_glitches, l1_glitches = glitch_sampler.glitches
    overlapping_h1, h1_glitches = split(h1_glitches, glitch_frac, axis=0)
    overlapping_l1, l1_glitches = torch.split(
        l1_glitches,
        [len(overlapping_h1), len(l1_glitches) - len(overlapping_h1)],
        dim=0,
    )
    overlapping = torch.stack([overlapping_h1, overlapping_l1], axis=1)

    # if we need to create duplicates of some of our
    # background to make this work, figure out how many
    num_h1, num_l1 = len(h1_glitches), len(l1_glitches)
    num_overlapping = len(overlapping)
    num_glitches = num_h1 + num_l1 + num_overlapping
    repeats = ceil(num_glitches / len(background))
    glitch_background = background.repeat(repeats, 1, 1)
    glitch_background = glitch_background[:num_glitches]

    # now insert the glitches
    start = h1_glitches.shape[-1] // 2 - kernel_size // 2
    stop = start + kernel_size
    glitch_background[:num_h1, 0] = h1_glitches[:, start:stop]
    glitch_background[num_h1 : num_h1 + num_l1, 1] = l1_glitches[:, start:stop]
    glitch_background[num_h1 + num_l1 :] = overlapping[:, :, start:stop]

    # finally create a tensor of background with waveforms injected
    waveforms, _ = waveform_sampler.sample(-1)
    repeats = ceil(len(waveforms) / len(background))
    waveform_background = background.repeat(repeats, 1, 1)
    waveform_background = waveform_background[: len(waveforms)]

    start = waveforms.shape[-1] // 2 - kernel_size // 2
    stop = start + kernel_size
    waveform_background += waveforms[:, :, start:stop]

    # concatenate everything into a single tensor
    # and create the associated labels
    X = torch.concat([background, glitch_background, waveform_background])
    y = torch.zeros((len(X),))
    y[-len(waveform_background) :] = 1

    logging.info("Performing validation on:")
    logging.info(f"    {len(background)} windows of background")
    logging.info(f"    {num_h1} H1 glitches")
    logging.info(f"    {num_l1} L1 glitches")
    logging.info(f"    {num_overlapping} overlapping glitches")
    logging.info(f"    {len(waveforms)} injected waveforms")
    return torch.utils.data.TensorDataset(X, y)


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
    augmentation_layers = OrderedDict()

    # build a glitch sampler from a pre-saved bank of
    # glitches which will randomly insert them into
    # either or both interferometer channels
    with h5py.File(glitch_dataset, "r") as f:
        h1_glitches = f["H1_glitches"][:]
        l1_glitches = f["L1_glitches"][:]

    if valid_frac is not None:
        h1_glitches, valid_h1_glitches = split(h1_glitches, 1 - valid_frac, 0)
        l1_glitches, valid_l1_glitches = split(l1_glitches, 1 - valid_frac, 0)
        valid_glitch_sampler = GlitchSampler(
            prob=1, max_offset=0, H1=valid_h1_glitches, L1=valid_l1_glitches
        )

    augmentation_layers["glitch_inserter"] = GlitchSampler(
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
            signals, valid_signals = split(signals, 1 - valid_frac, 0)
            valid_plus, valid_cross = valid_signals.transpose(1, 0, 2)

            slc = slice(-len(valid_signals), None)
            valid_injector = RandomWaveformInjection(
                dec=f["dec"][slc],
                psi=f["psi"][slc],
                phi=f["ra"][slc],  # no geocent_time recorded, so just use ra
                snr=lambda N: torch.ones((N,)) * 8,  # todo: pass mean_Snr
                sample_rate=sample_rate,
                highpass=highpass,
                trigger_offset=0,
                plus=valid_plus,
                cross=valid_cross,
            )

    plus, cross = signals.transpose(1, 0, 2)

    # instantiate source parameters as callable
    # distributions which will produce samples
    augmentation_layers["injector"] = RandomWaveformInjection(
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
    augmenter = MultiInputSequential(augmentation_layers)

    if valid_frac is not None:
        return augmenter, valid_glitch_sampler, valid_injector
    return augmenter, None, None


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
    **kwargs,
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
    augmenter, valid_glitch_sampler, valid_injector = prepare_augmentation(
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
        valid_injector.fit(H1=background[0], L1=background[1])

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
    preprocessor = Preprocessor(
        2,
        sample_rate,
        kernel_length,
        highpass=highpass,
        fduration=fduration,
    )

    # fit the whitening module to the background then
    # move eveyrthing to the desired device
    preprocessor.whitener.fit(background)
    preprocessor.whitener.to(device)

    # deterministic validation glitch sampler
    if valid_frac is not None:
        valid_dataset = make_validation_dataset(
            valid_background,
            valid_glitch_sampler,
            valid_injector,
            kernel_length,
            valid_stride,
            sample_rate,
            glitch_frac,
        )
        valid_dataset = torch.utils.data.DataLoader(
            valid_dataset,
            pin_memory=True,
            batch_size=batch_size * 16,
        )
    else:
        valid_dataset = None

    return train_dataset, valid_dataset, preprocessor
