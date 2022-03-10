import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

from bbhnet.data import dataloader


@pytest.fixture(scope="session")
def data_dir():
    data_dir = "tmp"
    os.makedirs(data_dir, exist_ok=True)
    yield Path(data_dir)
    shutil.rmtree(data_dir)


@pytest.fixture(params=[512, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture
def data_length():
    return 128


@pytest.fixture
def hanford_background(sample_rate, data_dir, data_length):
    x = np.random.randn(sample_rate * data_length)
    with h5py.File(data_dir / "hanford.h5", "w") as f:
        f["hoft"] = x
    return data_dir / "hanford.h5"


@pytest.fixture
def livingston_background(sample_rate, data_dir, data_length):
    x = np.random.randn(sample_rate * data_length)
    with h5py.File(data_dir / "livingston.h5", "w") as f:
        f["hoft"] = x
    return data_dir / "livingston.h5"


def test_random_waveform_dataset(
    hanford_background, livingston_background, sample_rate
):
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        hanford_background,
        livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        batches_per_epoch=10,
        device="cpu",
    )
    for i, (X, y) in enumerate(dataset):
        assert X.shape == (batch_size, 2, sample_rate)
        assert y.shape == (batch_size,)
        assert not y.any()

    assert i == 9


def test_random_waveform_dataset_whitening(
    hanford_background, livingston_background, sample_rate, data_length
):
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        hanford_background,
        livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        batches_per_epoch=10,
        device="cpu",
    )
    X = np.random.randn(32, 2, sample_rate)
    whitened = dataset.whiten(torch.Tensor(X)).numpy()

    asds = list(dataset.whitening_filter.numpy()[:, 0])
    for x, y in zip(X, whitened):
        for x_ifo, y_ifo, asd in zip(x, y, asds):
            ts = TimeSeries(x_ifo, dt=1 / sample_rate)
            fs = FrequencySeries(asd, df=1 / data_length)
            expected = ts.whiten(asd=fs)
            assert (expected == y_ifo).all()
