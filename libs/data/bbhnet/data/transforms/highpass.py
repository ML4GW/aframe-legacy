import math

import torch

from bbhnet.data.transforms.transform import Transform

# from torchaudio.functional import lfilter


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

        a_coeffs = torch.Tensor([a2, a1, a0])[None, None] / a0
        b_coeffs = torch.Tensor([b2, b1, b0])[None, None]
        self.a_coeffs = self.add_parameter(a_coeffs)
        self.b_coeffs = self.add_parameter(b_coeffs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.nn.functional.pad(X, [2, 0])
        X = torch.nn.function.conv1d(X, self.b_coeffs, groups=X.shape[1])
        X.div_(self.a_coeffs[:, :, -1])

        X = torch.nn.functional.pad(X, [0, 2])
        convd = torch.nn.functional.conv1d(X, self.a_coeffs, groups=X.shape[1])
        return X - convd
