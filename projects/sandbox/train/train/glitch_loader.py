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
        super().__init__()
        self.reads_per_chunk = reads_per_chunk
        self.chunks_per_epoch = chunks_per_epoch
        self.glitches_per_read = glitches_per_read
        self.ifo_files = ifo_files

        sizes = {}
        probs = {}
        self.ifos = list(ifo_files.keys())
        for ifo, fnames in self.ifo_files.items():
            sizes = []
            for f in fnames:
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
                    indices = np.sort(
                        np.random.permutation(num)[: self.glitches_per_read]
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
        reads_per_chunk: int,
        batches_per_chunk: int,
        chunks_per_epoch: int,
        glitches_per_read: int,
        glitches_per_batch: int,
        device: str,
        num_workers: Optional[int] = None,
        **ifo_paths: List[Path],
    ):
        super().__init__()
        self.batches_per_chunk = batches_per_chunk
        self.chunks_per_epoch = chunks_per_epoch
        self.glitches_per_batch = glitches_per_batch
        self.reads_per_chunk = reads_per_chunk
        self.glitches_per_read = glitches_per_read

        self.ifos = list(ifo_paths.keys())
        self.device = device
        self.num_workers = num_workers

        glitch_loader = GlitchLoader(
            glitches_per_read,
            reads_per_chunk,
            chunks_per_epoch,
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

    def __len__(self):
        if not self.num_workers:
            return self.chunks_per_epoch * self.batches_per_chunk

        num_chunks = (self.chunks_per_epoch - 1) // self.num_workers + 1
        return num_chunks * self.num_workers * self.batches_per_chunk

    def iter_epoch(self):
        num = self.reads_per_chunk * self.glitches_per_read
        for glitches in self.glitch_loader:
            for _ in range(self.batches_per_chunk):
                indices = torch.randint(num, size=(self.glitches_per_batch,))
                yield glitches[:, indices, :].to(self.device)

    def __iter__(self):
        return self.iter_epoch()
