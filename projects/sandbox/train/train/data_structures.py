from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

import h5py
import numpy as np
import torch

from ml4gw import gw
from ml4gw.distributions import PowerLaw
from ml4gw.transforms import SpectralDensity

if TYPE_CHECKING:
    from train.glitch_loader import ChunkedGlitchDataset


def sample_kernels(chunk, kernel_size, batch_size):
    max_idx = chunk.shape[-1] - kernel_size
    fname_idx = torch.randint(len(chunk), size=(batch_size, 1))
    start_idx = torch.randint(max_idx, size=(batch_size, 1))

    kernels = torch.arange(kernel_size).view(1, kernel_size)
    kernels = kernels.repeat(batch_size, 1)
    kernels += fname_idx * chunk.shape[-1] + start_idx

    return torch.take(chunk, kernels)


@dataclass
class ChunkedDataloader:
    """
    Iterable for generating batches of background data
    loaded on-the-fly from multiple HDF5 files. Loads
    `chunk_length`-sized randomly-sampled stretches of
    background from `reads_per_chunk` randomly sampled
    files up front, then samples `batches_per_chunk`
    batches of kernels from this chunk before loading
    in the next one. Terminates after `chunks_per_epoch`
    chunks have been exhausted, which amounts to
    `chunks_per_epoch * batches_per_chunk` batches.
    """

    fnames: List[str]
    ifos: List[str]
    kernel_length: float
    sample_rate: float
    batch_size: int
    reads_per_chunk: int
    chunk_length: float
    batches_per_chunk: int
    chunks_per_epoch: int
    device: str
    preprocessor: Optional[Callable] = None

    def __len__(self):
        return self.batches_per_chunk * self.chunks_per_epoch

    def __iter__(self):
        kernel_size = int(self.kernel_length * self.sample_rate)
        chunk_size = int(self.chunk_length * self.sample_rate)

        @torch.no_grad()
        def chunked_generator():
            shape = (self.reads_per_chunk, 1, chunk_size)
            chunks = [torch.zeros(shape) for _ in self.ifos]

            # initialize the batch tensor up front
            # so that all we have to do is populate
            # with data at each iteration
            batch_shape = (self.batch_size, len(self.ifos), kernel_size)
            batch = torch.zeros(batch_shape).to(self.device)

            for _ in range(self.chunks_per_epoch):
                # randomly select files to read random,
                # independently sampled chunks of data from
                for i in range(self.reads_per_chunk):
                    fname = np.random.choice(self.fnames)
                    with h5py.File(fname, "r") as f:
                        for j, ifo in enumerate(self.ifos):
                            dset = f[ifo]
                            start = np.random.choice(len(dset) - chunk_size)
                            x = dset[start : start + chunk_size]
                            chunks[j][i, 0] = torch.Tensor(x)

                # generate a batches from this chunk by
                # sampling from each IFO independently
                for _ in range(self.batches_per_chunk):
                    # sample kernels for each ifo and insert
                    # them into the batch
                    for i, chunk in enumerate(chunks):
                        kernels = sample_kernels(
                            chunk, kernel_size, self.batch_size
                        )
                        batch[:, i] = kernels

                    # generate y separately each time in
                    # case downstream augmentations update
                    # it in-place
                    y = torch.zeros((len(batch), 1), device=self.device)
                    if self.preprocessor is not None:
                        yield self.preprocessor(batch, y)
                    else:
                        yield batch, y

        return chunked_generator()


class PsdEstimator(torch.nn.Module):
    def __init__(
        self,
        background_length: float,
        sample_rate: float,
        fftlength: float,
        overlap: Optional[float] = None,
        average: str = "mean",
        fast: bool = True,
    ) -> None:
        super().__init__()
        self.background_size = int(background_length * sample_rate)
        self.spectral_density = SpectralDensity(
            sample_rate, fftlength, overlap, average, fast=fast
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        splits = [self.background_size, X.shape[-1] - self.background_size]
        background, X = torch.split(X, splits, dim=-1)
        psds = self.spectral_density(background.double())
        return X, psds


class ChannelSwapper(torch.nn.Module):
    """
    Data augmentation module that randomly swaps channels
    of a fraction of batch elements.

    Args:
        frac:
            Fraction of batch that will have channels swapped.
    """

    def __init__(self, frac: float = 0.5):
        super().__init__()
        self.frac = frac

    def forward(self, X):
        num = int(X.shape[0] * self.frac)
        indices = []
        if num > 0:
            num = num if not num % 2 else num - 1
            num = max(2, num)
            channel = torch.randint(X.shape[1], size=(num // 2,)).repeat(2)
            # swap channels from the first num / 2 elements with the
            # second num / 2 elements
            indices = torch.arange(num)
            target_indices = torch.roll(indices, shifts=num // 2, dims=0)
            X[indices, channel] = X[target_indices, channel]
        return X, indices


class ChannelMuter(torch.nn.Module):
    """
    Data augmentation module that randomly mutes 1 channel
    of a fraction of batch elements.

    Args:
        frac:
            Fraction of batch that will have channels muted.
    """

    def __init__(self, frac: float = 0.5):
        super().__init__()
        self.frac = frac

    def forward(self, X):
        num = int(X.shape[0] * self.frac)
        indices = []
        if num > 0:
            channel = torch.randint(X.shape[1], size=(num,))
            indices = torch.randint(X.shape[0], size=(num,))
            X[indices, channel] = torch.zeros(X.shape[-1], device=X.device)

        return X, indices


class SignalInverter(torch.nn.Module):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, X):
        if self.training:
            mask = torch.rand(size=X.shape[:-1]) < self.prob
            X[mask] *= -1
        return X


class SignalReverser(torch.nn.Module):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, X):
        if self.training:
            mask = torch.rand(size=X.shape[:-1]) < self.prob
            X[mask] = X[mask].flip(-1)
        return X


# TODO: use ml4gw version if/when it's merged
class SnrRescaler(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        waveform_duration: float,
        highpass: Optional[float] = None,
    ):
        super().__init__()
        self.highpass = highpass
        self.sample_rate = sample_rate
        self.df = 1 / waveform_duration
        waveform_size = int(waveform_duration * sample_rate)

        if highpass is not None:
            freqs = torch.fft.rfftfreq(waveform_size, 1 / sample_rate)
            self.register_buffer("mask", freqs >= highpass, persistent=False)
        else:
            self.mask = None

    def forward(
        self,
        responses: gw.WaveformTensor,
        asds: torch.Tensor,
        target_snrs: Optional[gw.ScalarTensor] = None,
    ):
        num_freqs = responses.shape[-1] // 2 + 1
        if asds.shape[-1] != num_freqs:
            asds = torch.nn.functional.interpolate(asds, size=(num_freqs,))
        snrs = gw.compute_network_snr(
            responses, asds**2, self.sample_rate, self.mask
        )
        if target_snrs is None:
            idx = torch.randperm(len(snrs))
            target_snrs = snrs[idx]

        target_snrs.to(snrs.device)
        weights = target_snrs / snrs
        rescaled_responses = responses * weights.view(-1, 1, 1)

        return rescaled_responses, target_snrs


class SnrSampler:
    def __init__(
        self,
        max_min_snr: float,
        min_min_snr: float,
        max_snr: float,
        alpha: float,
        decay_steps: int,
    ):
        self.max_min_snr = max_min_snr
        self.min_min_snr = min_min_snr
        self.max_snr = max_snr
        self.alpha = alpha
        self.decay_steps = decay_steps
        self._step = 0

        self.dist = PowerLaw(max_min_snr, max_snr, alpha)

    def __call__(self, N):
        return self.dist(N)

    def step(self):
        self._step += 1
        if self._step > self.decay_steps:
            return

        frac = self._step / self.decay_steps
        diff = self.max_min_snr - self.min_min_snr
        new = self.max_min_snr - frac * diff

        self.dist.x_min = new
        self.dist.normalization = new ** (-self.alpha + 1)
        self.dist.normalization -= self.max_snr ** (-self.alpha + 1)


class AframeDataloader:
    def __init__(
        self,
        background_loader: ChunkedDataloader,
        glitch_loader: "ChunkedGlitchDataset",
        augmentor: torch.nn.Module,
    ):

        self.background_loader = background_loader
        self.glitch_loader = glitch_loader
        self.augmentor = augmentor
        self.size = len(background_loader)

    def __len__(self):
        return self.size

    def __iter__(self):
        self.background_iter = iter(self.background_loader)
        self.glitch_iter = iter(self.glitch_loader)
        return self

    def __next__(self):
        X, y = next(self.background_iter)
        glitches = next(self.glitch_iter)
        glitches = glitches.transpose(1, 0)
        X, y = self.augmentor(X, glitches, y)
        return X, y
