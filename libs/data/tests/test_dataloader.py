import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
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


@pytest.fixture(params=[0.01, 0.1, 0.5, 0.9])
def glitch_frac(request):
    return request.param


@pytest.fixture(params=[2, 4])
def glitch_length(request):
    return request.param


def write_timeseries(ts, fname, data_dir):
    with h5py.File(data_dir / fname, "w") as f:
        f["hoft"] = ts
    return data_dir / fname


@pytest.fixture
def random_hanford_background(sample_rate, data_dir, data_length):
    x = np.random.randn(sample_rate * data_length)
    return write_timeseries(x, "hanford.h5", data_dir)


@pytest.fixture
def random_livingston_background(sample_rate, data_dir, data_length):
    x = np.random.randn(sample_rate * data_length)
    return write_timeseries(x, "livingston.h5", data_dir)


@pytest.fixture
def sequential_hanford_background(sample_rate, data_dir, data_length):
    x = np.arange(sample_rate * data_length)
    return write_timeseries(x, "hanford.h5", data_dir)


@pytest.fixture
def sequential_livingston_background(sample_rate, data_dir, data_length):
    x = np.arange(sample_rate * data_length)
    return write_timeseries(x, "livingston.h5", data_dir)


@pytest.fixture
def zeros_hanford_background(sample_rate, data_dir, data_length):
    x = np.zeros((sample_rate * data_length,))
    return write_timeseries(x, "hanford.h5", data_dir)


@pytest.fixture
def zeros_livingston_background(sample_rate, data_dir, data_length):
    x = np.zeros((sample_rate * data_length,))
    return write_timeseries(x, "livingston.h5", data_dir)


@pytest.fixture
def negative_glitches(sample_rate, data_dir, glitch_length):
    num_glitches = 128
    x = -np.arange(sample_rate * glitch_length * num_glitches) - 1
    x = x.reshape(num_glitches, -1)

    with h5py.File(data_dir / "glitches.h5", "w") as f:
        f["hanford"] = x
        f["livingston"] = x
    return data_dir / "glitches.h5"


def test_random_waveform_dataset(
    sequential_hanford_background,
    sequential_livingston_background,
    sample_rate,
):
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        sequential_hanford_background,
        sequential_livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        batches_per_epoch=10,
        device="cpu",
    )

    # test the background sampling method to make sure
    # that the base batch is generated properly
    X = dataset.sample_from_background(independent=True)

    # make sure that we're not sampling in order
    assert not (np.diff(X[:, 0, 0].numpy()) == 1).all()

    # now make sure each timeseries is sequential
    # and that the two interferometers don't match
    for x in X:
        assert not (x[0] == x[1]).all()
        for x_ifo in x:
            assert (np.diff(x_ifo.numpy()) == 1).all()

    # make sure if we're sampling non-independently
    # that the data from both interferometers matches
    X = dataset.sample_from_background(independent=False)
    for x in X:
        assert (x[0] == x[1]).all()

    # now go through and make sure that the iteration
    # method generates data of the right size
    for i, (X, y) in enumerate(dataset):
        assert X.shape == (batch_size, 2, sample_rate)
        assert y.shape == (batch_size,)
        assert not y.any()

    assert i == 9


def test_random_waveform_dataset_whitening(
    random_hanford_background,
    random_livingston_background,
    sample_rate,
    data_length,
):
    """
    Test the `.whiten` method to make sure that it
    produces results roughly consistent with gwpy's
    whitening functionality using the background
    data's ASD
    """

    # create a dataset from the background
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        random_hanford_background,
        random_livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        batches_per_epoch=10,
        device="cpu",
    )

    # whiten a random batch of data manually
    X = np.random.randn(32, 2, sample_rate)
    whitened = dataset.whiten(torch.Tensor(X)).numpy()

    # for each sample, whiten the sample using the
    # corresponding background ASD with gwpy and
    # ensure that the results are reasonably close
    for x, w in zip(X, whitened):
        for i, bkgrd in enumerate(
            [dataset.hanford_background, dataset.livingston_background]
        ):
            bkgrd_ts = TimeSeries(bkgrd.numpy(), dt=1 / sample_rate)
            bkgrd_asd = bkgrd_ts.asd(fftlength=2)
            ts = TimeSeries(x[i], dt=1 / sample_rate)
            ts = ts.whiten(asd=bkgrd_asd).value

            # make sure that the relative error from the two
            # whitening methods is in the realm of the reasonable.
            # We could make these bounds tighter for most sample
            # rates of interest, but the 512 just has too much noise
            err = np.abs(ts - w[i]) / np.abs(ts)
            assert np.percentile(err, 80) < 0.02
            assert np.percentile(err, 95) < 0.1


def test_glitch_sampling(
    zeros_hanford_background,
    zeros_livingston_background,
    negative_glitches,
    glitch_frac,
    sample_rate,
):
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        zeros_hanford_background,
        zeros_livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        glitch_frac=glitch_frac,
        glitch_dataset=negative_glitches,
        batches_per_epoch=10,
        device="cpu",
    )
    expected_num = int(glitch_frac * batch_size)
    if expected_num == 0:
        expected_num = 1
    assert dataset.num_glitches == expected_num

    for X, _ in dataset:
        for i in range(dataset.num_glitches):
            x = X[i].numpy()
            assert (np.diff(x[0]) < 0).all() or (np.diff(x[1]) < 0).all()
        for i in range(dataset.num_glitches, batch_size):
            x = X[i].numpy()
            assert not ((np.diff(x[0]) < 0).all() or (np.diff(x[1]) < 0).all())
