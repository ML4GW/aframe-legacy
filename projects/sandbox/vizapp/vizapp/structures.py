from typing import Optional

import torch

from aframe.architectures.preprocessor import Whitener
from ml4gw.transforms import SpectralDensity
from ml4gw.utils.slicing import unfold_windows


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


class Preprocessor(torch.nn.Module):
    def __init__(
        self,
        whitener: Whitener,
        psd_estimator: PsdEstimator,
    ) -> None:
        super().__init__()
        self.whitener = whitener
        self.psd_estimator = psd_estimator

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X, psds = self.psd_estimator(X)
        X = self.whitener(X, psds)
        return X


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
        x = self.whitener(x, psd)
        windows = unfold_windows(x, self.kernel_size, self.stride_size)
        return windows[:, 0], x
