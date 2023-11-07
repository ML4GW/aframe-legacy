from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import torch


class Hdf5GlitchDataset(torch.utils.data.IterableDataset):
    """
    Iterable dataset that samples and loads batches of
    glitches from a sets of HDF5 files.
    It is _strongly_ recommended that these files have been
    written using [chunked storage]
    (https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage).
    This has shown to produce increases in read-time speeds
    of over an order of magnitude.

    Args:
        batch_size:
            Number of glitches to sample at each iteration.
        batches_per_epoch:
            Number of batches to generate during each call
            to `__iter__`.
        files:
            Dictionary mapping dataset names to lists of
            paths to HDF5 files containing those datasets.
    """

    def __init__(
        self,
        batch_size: int,
        batches_per_epoch: int,
        files: Dict[str, List[Path]],
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.files = files
        self.channels = list(files.keys())

        sizes = {}
        probs = {}

        # get sizes of each file for each channel
        for channel, fnames in self.files.items():
            sizes = []
            for f in fnames:
                with h5py.File(f, "r") as f:
                    size = len(f["glitches"])
                    sizes.append(size)

                    # infer kernel length
                    self.kernel_length = f["glitches"].shape[-1]

            total = sum(sizes)
            probs[channel] = np.array([i / total for i in sizes])
        self.probs = probs

    def __len__(self) -> int:
        return self.batches_per_epoch

    def sample_fnames(self):
        fnames = {}
        for channel, files in self.files.items():
            fnames[channel] = np.random.choice(
                files,
                p=self.probs[channel],
                size=(self.batch_size,),
                replace=True,
            )
        return fnames

    def sample_batch(self):
        # allocate memory up front
        x = np.zeros((self.batch_size, len(self.channels), self.kernel_length))

        # for each channel, sample files for each batch element
        fnames = self.sample_fnames()

        for i, files in enumerate(fnames.values()):
            unique_files, inv = np.unique(files, return_inverse=True)

            for j, file in enumerate(unique_files):
                batch_indices = np.where(inv == j)[0]

                with h5py.File(file, "r") as f:
                    num = len(f["glitches"])
                    indices = np.random.randint(num, size=len(batch_indices))
                    for b, idx in zip(batch_indices, indices):
                        x[b, i] = f["glitches"][idx]
        return torch.Tensor(x)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_batches = self.batches_per_epoch
        else:
            num_batches, remainder = divmod(
                self.batches_per_epoch, worker_info.num_workers
            )
            if worker_info.id < remainder:
                num_batches += 1

        for _ in range(num_batches):
            yield self.sample_batch()


class ChunkedGlitchSampler(torch.utils.data.IterableDataset):
    """
    Wrapper dataset that will loop through chunks of glitches
    produced from another iterable and randomly sample batches.

    Args:
        chunk_it:
            Iterator that will produce chunks of glitches to sample from.
            Each chunk should have shape
            `(N, C, T)` where `N` is the number of chunks
            to sample from, `C` is the number of channels,
            and `T` is the number of samples along the
            time dimension for each chunk.
        batch_size:
            Number of windows to sample at each iteration
        batches_per_chunk:
            Number of batches of windows to sample from
            each chunk before moving on to the next one.
            Sampling fewer batches from each chunk means
            a lower likelihood of sampling duplicate windows,
            but an increase in chunk-loading overhead.
        device:
            Which device chunks should be moved to upon loading.

    """

    def __init__(
        self,
        chunk_it: Iterable,
        batch_size: int,
        batches_per_chunk: int,
        device: str = "cpu",
    ):

        super().__init__()
        self.chunk_it = chunk_it
        self.batch_size = batch_size
        self.batches_per_chunk = batches_per_chunk
        self.device = device

    def __len__(self):
        return len(self.chunk_it) * self.batches_per_chunk

    def __iter__(self):
        # initiate first chunk
        it = iter(self.chunk_it)
        chunk = next(it)
        num = chunk.shape[0]

        # continuoulsy produce batches from chunks
        # until chunk iterator is exhausted
        while True:

            # yield batches_per_chunk batches from
            # current chunk
            for _ in range(self.batches_per_chunk):
                indices = torch.randint(num, size=(self.batch_size,))
                yield chunk[indices, :, :].to(self.device)

            # move onto next chunk if possible
            # otherwise break the epoch
            try:
                chunk = next(it)
            except StopIteration:
                break
            else:
                pass
