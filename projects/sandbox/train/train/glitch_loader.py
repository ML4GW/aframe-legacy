from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch


class GlitchLoader(torch.utils.data.IterableDataset):
    def __init__(
        self,
        glitches_per_read: int,
        reads_per_chunk: int,
        chunks_per_epoch: int,
        **ifo_files: List[Path],
    ) -> None:
        self.reads_per_chunk = reads_per_chunk
        self.chunks_per_epoch = chunks_per_epoch
        self.glitches_per_read = glitches_per_read
        self.ifo_files = ifo_files

        sizes = {}
        probs = {}
        for ifo, fnames in self.ifo_files.items():
            sizes = []
            for f in self.fnames:
                with h5py.File(f, "r") as f:
                    size = len(f["glitches"])
                    sizes.append(size)

            total = sum(sizes)
            probs[ifo] = np.array([i / total for i in sizes])

        self.probs = probs

    def sample_fnames(self):
        sampled = {}
        for ifo, files in self.ifo_files.items():
            sampled[ifo] = np.random.choice(
                files,
                p=self.probs[ifo],
                size=(self.reads_per_chunk,),
                replace=True,
            )
        return sampled

    def load(self):
        fnames = self.sample_fnames()
        glitches = []
        for ifo in self.ifos:
            ifo_glitches = []
            ifo_files = fnames[ifo]
            for fname in ifo_files:
                with h5py.File(fname, "r") as f:
                    num = len(f["glitches"])
                    indices = torch.randint(
                        num, size=(self.glitches_per_read,)
                    )
                    ifo_glitches.extend(f["glitches"][indices])
            glitches.append(ifo_glitches)

        return np.stack(glitches)

    def iter_epoch(self):
        for _ in range(self.chunks_per_epoch):
            yield torch.Tensor(self.load())

    def collate(self, xs):
        return torch.cat(xs, axis=0)

    def __iter__(self):
        return self.iter_epoch()


class ChunkedGlitchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        reads_per_chunk: int,  # number of files to read per chunk
        batches_per_chunk: int,  # number of batches before loading new chunk
        glitches_per_read: int,  # number of glitches to read per file
        glitches_per_batch: int,  # glitch_prob * batch_size
        device: str,
        num_workers: Optional[int] = None,
        **ifo_paths: List[Path],
    ):
        self.batches_per_chunk = batches_per_chunk
        self.glitches_per_batch = glitches_per_batch
        self.reads_per_chunk = reads_per_chunk
        self.glitches_per_read = glitches_per_read

        self.ifos = list(ifo_paths.keys())
        self.device = device
        self.num_workers = num_workers

        glitch_loader = GlitchLoader(
            glitches_per_read,
            reads_per_chunk,
            batches_per_chunk,
            **ifo_paths,
        )

        if not num_workers:
            self.glitch_loader = glitch_loader
        else:
            self.glitch_loader = torch.utils.data.DataLoader(
                glitch_loader,
                batch_size=num_workers,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=glitch_loader.collate,
            )

    def iter_epoch(self):
        for glitches in self.glitch_loader:
            for _ in range(self.batches_per_chunk):
                indices = torch.randint(
                    len(glitches),
                    size=(self.glitches_per_batch,),
                )
                yield glitches[indices].to(self.device)

    def __iter__(self):
        self.iter_epoch()
