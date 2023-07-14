from typing import List, Optional

import h5py
import numpy as np
import torch


class GlitchLoader(torch.utils.data.IterableDataset):
    def __init__(
        self,
        fnames: List[str],
        glitches_per_read: int,
        reads_per_chunk: int,
        chunks_per_epoch: int,
        ifos: List[str],
    ) -> None:
        self.fnames = fnames
        self.reads_per_chunk = reads_per_chunk
        self.chunks_per_epoch = chunks_per_epoch
        self.ifos = ifos

        sizes = []
        for f in self.fnames:
            with h5py.File(f, "r") as f:
                size = len(f[self.ifos[0]])
                sizes.append(size)

        self.glitches_per_read = min(glitches_per_read, min(sizes))
        total = sum(sizes)
        self.probs = np.array([i / total for i in sizes])

    def sample_fnames(self):
        return np.random.choice(
            self.fnames,
            p=self.probs,
            size=(self.reads_per_chunk,),
            replace=True,
        )

    def load(self):
        fnames = self.sample_fnames()
        glitches = []
        for fname in fnames:
            with h5py.File(fname, "r") as f:
                for ifo in self.ifos:
                    indices = torch.randint(self.glitches_per_read)
                    glitches.append(f[ifo][indices])

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
        fnames: List[str],
        ifos: List[str],
        reads_per_chunk: int,  # number of files to read per chunk
        batches_per_chunk: int,  # number of batches before loading new chunk
        glitches_per_read: int,  # number of glitches to read per file
        glitches_per_batch: int,  # glitch_prob * batch_size
        device: str,
        num_workers: Optional[int] = None,
    ):
        self.fnames = fnames
        self.batches_per_chunk = batches_per_chunk
        self.glitches_per_batch = glitches_per_batch
        self.reads_per_chunk = reads_per_chunk
        self.glitches_per_read = glitches_per_read
        self.ifos = ifos

        self.device = device
        self.num_workers = num_workers

        glitch_loader = GlitchLoader(
            fnames,
            glitches_per_read,
            reads_per_chunk,
            batches_per_chunk,
            ifos,
        )

        if not num_workers:
            self.glitch_loader = glitch_loader
        else:
            self.chunk_loader = torch.utils.data.DataLoader(
                fnames,
                glitches_per_read,
                reads_per_chunk,
                batches_per_chunk,
                ifos,
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
