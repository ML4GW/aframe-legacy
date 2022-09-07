import math

import torch
from torchaudio.functional import lfilter

from bbhnet.data.transforms.transform import Transform


class HighpassFilter(Transform):
    def __init__(
        self, cutoff: float, sample_rate: float, Q: float = 1 / math.sqrt(2)
    ) -> None:
        super().__init__()
        w0 = 2 * math.pi * cutoff / sample_rate
        alpha = math.sin(w0) / 2.0 / Q

        b0 = (1 + math.cos(w0)) / 2
        b1 = -1 - math.cos(w0)
        b2 = b0
        a0 = 1 + alpha
        a1 = -2 * math.cos(w0)
        a2 = 1 - alpha

        self.a_coeffs = self.add_parameter([a0, a1, a2])
        self.b_coeffs = self.add_parameter([b0, b1, b2])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return lfilter(X, self.a_coeffs, self.b_coeffs)
