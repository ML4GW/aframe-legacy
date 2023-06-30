from typing import Optional, Tuple

import torch

from aframe.architectures.preprocessor import Whitener
from ml4gw.transforms import SpectralDensity
from ml4gw.utils.slicing import unfold_windows


class BackgroundSnapshotter(torch.nn.Module):
    def __init__(
        self,
        psd_length,
        kernel_length,
        fduration,
        sample_rate,
        inference_sampling_rate,
    ) -> None:
        super().__init__()
        state_length = kernel_length + fduration + psd_length
        state_length -= 1 / inference_sampling_rate
        self.state_size = int(state_length * sample_rate)

    def forward(
        self, update: torch.Tensor, snapshot: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        x = torch.cat([snapshot, update], axis=-1)
        snapshot = x[:, :, -self.state_size :]
        return x, snapshot


class BatchWhitener(torch.nn.Module):
    def __init__(
        self,
        kernel_length: float,
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        fduration: float,
        fftlength: float = 2,
        highpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.stride_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate)

        length = (batch_size - 1) / inference_sampling_rate
        length += kernel_length + fduration
        self.size = int(length * sample_rate)

        self.spectral_density = SpectralDensity(
            sample_rate,
            fftlength=fftlength,
            overlap=None,
            average="mean",
            fast=highpass is not None,
        )
        self.whitener = Whitener(fduration, sample_rate, highpass)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits = [x.shape[-1] - self.size, self.size]
        background, x = torch.split(x, splits, dim=-1)
        psd = self.spectral_density(background.double())
        x = self.whitener(x.double(), psd)
        x = unfold_windows(x, self.kernel_size, self.stride_size)
        return x[:, 0]
