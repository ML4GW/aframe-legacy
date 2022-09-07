import numpy as np
import torch

from bbhnet.data.transforms import Transform
from ml4gw.utils.slicing import sample_kernels


# TODO: generalize to arbitrary ifos
class GlitchSampler(Transform):
    def __init__(
        self, prob: float, max_offset: int, **glitches: np.ndarray
    ) -> None:
        super().__init__()
        glitches = torch.nn.ParameterList()
        for ifo in glitches:
            param = self.add_parameter(ifo)
            glitches.append(param)

        self.prob = prob
        self.max_offset = max_offset

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if X.shape[1] < len(self.glitches):
            raise ValueError(
                "Can't insert glitches into tensor with {} channels "
                "using glitches from {} ifos".format(
                    X.shape[1], len(self.glitches)
                )
            )

        for i, ifo in enumerate(self.glitches):
            mask = torch.rand(size=X.shape[:1]) < self.prob
            N = mask.sum().item()
            idx = torch.randint(len(X), size=(N,))

            glitches = ifo[idx]
            glitches = sample_kernels(
                glitches,
                kernel_size=X.shape[-1],
                max_center_offset=self.max_offset,
                coincident=False,
            )
            X[mask] = glitches
        return X, y
