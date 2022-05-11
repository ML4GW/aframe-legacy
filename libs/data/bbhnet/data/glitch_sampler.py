from typing import List

import h5py
import numpy as np
import torch

from bbhnet.data.utils import sample_kernels


class GlitchSampler:
    def __init__(
        self, glitch_dataset: str, ifos: List[str], device: str
    ) -> None:
        # TODO: will these need to be resampled?
        self.ifos = ifos
        self.n_ifos = len(ifos)

        self.glitches = {}
        with h5py.File(glitch_dataset, "r") as f:
            for ifo in ifos:
                self.glitches[ifo] = f[f"{ifo}_glitches"][:]
                self.glitches[ifo] = torch.Tensor(self.glitches[ifo]).to(
                    device
                )

    def sample(self, N: int, size: int) -> np.ndarray:

        glitch_count = N
        sampled_glitches = {}

        for i, ifo in enumerate(self.ifos):

            # if on the last iteration
            # sample enough glitches for this ifo to
            # satisfy user request
            if i == (len(self.ifos) - 1):
                num = glitch_count

            # otherwise choose random number of glitches
            else:
                num = np.random.randint(glitch_count)

            if num > 0:
                sampled_glitches[ifo] = sample_kernels(
                    self.glitches[ifo], size, num
                )
                sampled_glitches[ifo] = torch.stack(sampled_glitches, axis=0)
            else:
                sampled_glitches = None

            # update number of glitches left to sample
            glitch_count -= num

        return sampled_glitches
