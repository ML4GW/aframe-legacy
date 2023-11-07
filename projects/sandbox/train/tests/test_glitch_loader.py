from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import torch
from train.glitch_loader import ChunkedGlitchSampler, Hdf5GlitchDataset


class TestHdf5GlitchDataset:
    @pytest.fixture
    def channels(self):
        return ["A", "B"]

    @pytest.fixture
    def sample_rate(self):
        return 128

    @pytest.fixture
    def kernel_size(self, sample_rate):
        return 2 * sample_rate

    @pytest.fixture
    def batch_size(self):
        return 128

    @pytest.fixture
    def batches_per_epoch(self):
        return 10

    @pytest.fixture
    def fnames(self, channels, kernel_size, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        files = {}
        fnames = {"a.h5": 15, "b.h5": 5, "c.h5": 15, "d.h5": 5}
        keys = sorted(fnames)
        for channel in channels:
            idx = 0
            sub_dir = data_dir / channel
            sub_dir.mkdir(exist_ok=True, parents=True)
            files[channel] = [sub_dir / fname for fname in keys]
            for fname in keys:
                num = fnames[fname]
                with h5py.File(sub_dir / fname, "w") as f:
                    x = (
                        np.arange(idx, idx + num)
                        .repeat(kernel_size)
                        .reshape(-1, kernel_size)
                    )
                    f["glitches"] = x
                    idx += num
        return files

    @pytest.fixture
    def dataset(
        self,
        fnames,
        batch_size,
        batches_per_epoch,
    ):
        return Hdf5GlitchDataset(batch_size, batches_per_epoch, fnames)

    def test_init(self, dataset):
        assert len(dataset) == 10
        assert dataset.channels == ["A", "B"]
        expected_probs = np.array([0.375, 0.125, 0.375, 0.125])
        for channel in dataset.channels:
            np.testing.assert_equal(expected_probs, dataset.probs[channel])

    def test_sample_fnames(self, dataset):
        fnames = dataset.sample_fnames()[dataset.channels[0]]
        assert len(fnames) == 128

        # really weak check: let's at least confirm
        # that we sample the 20 glitch file more than
        # we sample the 10 glitch file.
        counts = {
            fname.name: 0 for fname in dataset.files[dataset.channels[0]]
        }
        for _ in range(10):
            fnames = dataset.sample_fnames()[dataset.channels[0]]
            for fname in fnames:
                counts[fname.name] += 1
        assert counts["a.h5"] > counts["b.h5"]

    def test_sample_batch(self, dataset, kernel_size, fnames):
        x = dataset.sample_batch()
        assert x.shape == (128, 2, kernel_size)

        # mock fnames to choose glitches from a.h5 file only
        fnames = {
            channel: [fnames[channel][0] for _ in range(128)]
            for channel in dataset.channels
        }
        dataset.sample_fnames = MagicMock(return_value=fnames)

        x = dataset.sample_batch()
        assert x.shape == (128, 2, kernel_size)
        # all should come from first file
        # so should be less than 15
        assert np.all(x.numpy() < 15)

    def test_iter(self, dataset, kernel_size):
        for i, x in enumerate(dataset):
            assert x.shape == (128, 2, kernel_size)
        assert i == 9


class TestChunkedGlitchSampler:
    @pytest.fixture
    def chunks_per_epoch(self):
        return 6

    @pytest.fixture
    def glitches_per_chunk(self):
        return 6

    @pytest.fixture
    def sample_rate(self):
        return 128

    @pytest.fixture
    def kernel_size(self, sample_rate):
        return 2 * sample_rate

    @pytest.fixture
    def chunk_it(self, chunks_per_epoch, glitches_per_chunk, kernel_size):
        def it():
            for i in range(chunks_per_epoch):
                chunk = []
                idx = 0
                for _ in range(6):
                    x = (
                        torch.arange(idx, idx + glitches_per_chunk)
                        .repeat(kernel_size)
                        .reshape(-1, kernel_size)
                    )
                    row = torch.stack([x, -x], axis=1)
                    chunk.append(row)
                chunk = torch.cat(chunk)

                yield chunk

        return it

    @pytest.fixture
    def batch_size(self):
        return 8

    @pytest.fixture
    def batches_per_chunk(self):
        return 7

    @pytest.fixture
    def dataset(
        self,
        chunk_it,
        batch_size,
        batches_per_chunk,
    ):
        return ChunkedGlitchSampler(
            chunk_it(),
            batch_size=batch_size,
            batches_per_chunk=batches_per_chunk,
            device="cpu",
        )

    def test_iter(self, dataset):
        for i, x in enumerate(dataset):
            assert x.shape == (8, 2, 256)

        assert i == 41
