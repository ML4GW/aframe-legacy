from typing import Optional

import h5py
import numpy as np
import torch

from bbhnet.data.utils import sample_kernels


# TODO: generalize to arbitrary ifos
class GlitchSampler:
    def __init__(
        self,
        glitch_dataset: str,
        deterministic: bool = False,
        frac: Optional[float] = None,
    ) -> None:

        self.glitches = {}
        with h5py.File(glitch_dataset, "r") as f:
            self.ifos = list(f.keys())
            for ifo in self.ifos:
                ifo_glitches = f[ifo][:]

                if frac is not None:
                    num_glitches = int(frac * len(ifo_glitches))
                    if frac < 0:
                        ifo_glitches = ifo_glitches[num_glitches:]

                    else:
                        ifo_glitches = ifo_glitches[:num_glitches]

                self.glitches[ifo] = torch.Tensor(ifo_glitches)

        self.deterministic = deterministic

    def to(self, device: str) -> None:
        for ifo in self.ifos:
            self.glitches[ifo] = self.glitches[ifo].to(device)

    def sample(self, N: int, size: int, offset: int = 0) -> np.ndarray:
        """Sample glitches from each interferometer

        If `self.deterministic` is `True`, this will grab the first
        `N` glitches from each interferometer, with the center of
        each kernel placed at the trigger time minus some specified
        amount of offset.

        If `self.deterministic` is `False`, this will sample _at most_
        `N` kernels from each interferometer, with the _total_ glitches
        sampled equal to `N`. The sampled glitches will be chosen at
        random, and the kernel sampled from each glitch will be randomly
        selected, with `offset` indicating the maximum distance the right
        edge of the kernel can be from the trigger time, i.e. the default
        value of 0 indicates that every kernel must contain the trigger.
        """

        sampled_glitches = {}
        if self.deterministic:
            if N == -1:
                N = len(self.hanford)

            center = int(self.glitches[self.ifos[0]].shape[-1] // 2)
            left = int(center + offset - size // 2)
            right = int(left + size)

            for ifo in self.ifos:
                sampled_glitches[ifo] = self.glitches[ifo][:N, left:right]
        else:
            for i, ifo in enumerate(self.ifos):

                # if on the last ifo
                # set num to remaining requested
                if i == len(self.ifos) - 1:
                    num = N

                # otherwise choose random num
                # from remaining gliches requested
                else:
                    num = np.random.randint(N)

                # update N to reflect num
                # of glitches sampled already
                N = N - num

                # sample glitches
                if num > 0:
                    sampled_glitches[ifo] = sample_kernels(
                        self.glitches[ifo], size, offset, num
                    )
                    sampled_glitches[ifo] = torch.stack(
                        sampled_glitches[ifo], axis=0
                    )
                else:
                    sampled_glitches[ifo] = None

        return sampled_glitches
