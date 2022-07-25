import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from gwpy.frequencyseries import FrequencySeries

from bbhnet.data import dataloader
from bbhnet.data.glitch_sampler import GlitchSampler
from bbhnet.data.waveform_sampler import WaveformSampler


def mock_asd(data_length, sample_rate):
    # mock asd object to ones
    # so that the ValueError doesn't
    # get raised  for 0 asd values
    df = 1 / data_length
    fmax = sample_rate / 2
    nfreqs = int(fmax / df)
    asd = FrequencySeries(np.ones(nfreqs), df=df, channel="H1:STRAIN")
    return asd


@pytest.fixture
def data_length():
    return 128


@pytest.fixture(params=[0.01, 0.1, 0.5, 0.9])
def glitch_frac(request):
    return request.param


@pytest.fixture(params=[0.01, 0.1, 0.5, 0.9])
def waveform_frac(request):
    return request.param


@pytest.fixture
def t0():
    return 1234567890


@pytest.fixture
def data_size(sample_rate, data_length):
    return int(sample_rate * data_length)


@pytest.fixture
def ones_data(data_size):
    return np.ones((data_size,))


@pytest.fixture
def sequential_data(data_size):
    return np.arange(data_size)


@pytest.fixture
def write_background(write_timeseries, t0, data_dir):
    def f(fname, x):
        write_timeseries(fname, hoft=x, t0=t0)
        return data_dir / fname

    return f


@pytest.fixture
def sequential_hanford_background(sequential_data, write_background):
    return write_background("hanford.h5", sequential_data)


@pytest.fixture
def sequential_livingston_background(sequential_data, write_background):
    return write_background("livingston.h5", -sequential_data)


@pytest.fixture
def ones_hanford_background(ones_data, write_background):
    return write_background("hanford.h5", ones_data)


@pytest.fixture
def ones_livingston_background(ones_data, write_background):
    return write_background("livingston.h5", ones_data)


@pytest.fixture(params=["path", "sampler"])
def glitch_sampler(arange_glitches, request):
    if request.param == "path":
        return arange_glitches
    else:
        return GlitchSampler(arange_glitches)


def validate_sequential(X):
    # make sure that we're not sampling in order
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    assert not (np.diff(X[:, 0, 0]) == 1).all()

    # now make sure each timeseries is sequential
    # and that the two interferometers don't match
    for x in X:
        assert not (x[0] == x[1]).all()
        for i, x_ifo in enumerate(x):
            diff = (-1) ** i
            assert (np.diff(x_ifo) == diff).all()


def validate_speed(dataset, N, limit):
    dataset.batches_per_epoch = N
    start_time = time.time()
    for _ in dataset:
        continue
    delta = time.time() - start_time
    assert delta / N < limit


def validate_dataset(dataset, cutoff_idx, target):
    # TODO: include "test_sample: Callable" argument
    # for testing each individual X, y sample that
    # isn't background. Check that the center is in
    # and for sines check that the wave has a valid freq
    for X, y in dataset:
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        for i, x in enumerate(X):
            # check to make sure ifo is not all 0s
            is_background = (x == 1).all()

            if target:
                # y == 1 targets should come at the end
                not_background = i >= cutoff_idx
                assert y[i] == int(not is_background)
            else:
                # y == 0 targets that _aren't_ background
                # should come at the start
                not_background = i < cutoff_idx

            # make sure you're either in the position
            # expected for a non-background of the
            # indicated target type, or you're background
            assert not_background ^ is_background, (i, x)


# Test the training dataset class first
def test_random_waveform_dataset(
    sequential_hanford_background,
    sequential_livingston_background,
    sample_rate,
    data_size,
    device,
):
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        sequential_hanford_background,
        sequential_livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        batches_per_epoch=10,
    )

    for ifo in ["hanford", "livingston"]:
        background = getattr(dataset, f"{ifo}_background")
        assert background.device.type == "cpu"
        assert background.dtype == torch.float64

    dataset.to(device)
    for ifo in ["hanford", "livingston"]:
        background = getattr(dataset, f"{ifo}_background")
        assert background.device.type == device
        assert background.dtype == torch.float32

    # test the background sampling method to make sure
    # that the base batch is generated properly
    X = dataset.sample_from_background().cpu().numpy()
    validate_sequential(X)

    # now go through and make sure that the iteration
    # method generates data of the right size
    for i, (X, y) in enumerate(dataset):
        assert X.shape == (batch_size, 2, sample_rate)
        validate_sequential(X)

        # make sure targets are all 0 because we
        # have no waveform sampling
        assert y.shape == (batch_size,)
        assert not y.cpu().numpy().any()

    assert i == 9


def test_random_waveform_dataset_with_glitch_sampling(
    ones_hanford_background,
    ones_livingston_background,
    glitch_sampler,
    glitch_frac,
    sample_rate,
    data_length,
    device,
):
    batch_size = 32
    dataset = dataloader.RandomWaveformDataset(
        ones_hanford_background,
        ones_livingston_background,
        kernel_length=1,
        sample_rate=sample_rate,
        batch_size=batch_size,
        glitch_frac=glitch_frac,
        glitch_sampler=glitch_sampler,
        batches_per_epoch=10,
    )
    dataset.to(device)
    assert dataset.glitch_sampler.hanford.device.type == device
    assert dataset.glitch_sampler.livingston.device.type == device

    expected_num = max(1, int(glitch_frac * batch_size))
    assert dataset.num_glitches == expected_num
    validate_dataset(dataset, dataset.num_glitches, 0)

    if device == "cpu":
        return
    validate_speed(dataset, N=100, limit=0.05)


def test_random_waveform_dataset_with_waveform_sampling(
    ones_hanford_background,
    ones_livingston_background,
    sine_waveforms,
    waveform_frac,
    sample_rate,
    data_length,
    device,
):
    waveform_sampler = WaveformSampler(
        sine_waveforms, sample_rate, min_snr=20, max_snr=40
    )

    batch_size = 32
    with patch(
        "gwpy.timeseries.TimeSeries.asd",
        return_value=mock_asd(data_length, sample_rate),
    ) as mock:
        dataset = dataloader.RandomWaveformDataset(
            ones_hanford_background,
            ones_livingston_background,
            kernel_length=1,
            sample_rate=sample_rate,
            batch_size=batch_size,
            waveform_sampler=waveform_sampler,
            waveform_frac=waveform_frac,
            batches_per_epoch=10,
        )
        dataset.to(device)

    # TODO: test that we don't need to be fit
    # if the waveform sampler has already been fit
    mock.assert_called()
    expected_num = max(1, int(waveform_frac * batch_size))
    assert dataset.num_waveforms == expected_num

    # if the dataset is going to request more waveforms
    # than we have, a ValueError should get raised
    if dataset.num_waveforms > 10:
        with pytest.raises(ValueError):
            next(iter(dataset))
        return
    validate_dataset(dataset, batch_size - dataset.num_waveforms, 1)

    if device == "cpu":
        return
    validate_speed(dataset, N=100, limit=0.05)


# now run the same tests for the deterministic validation sampler
@pytest.fixture(params=[0.1, 0.5, 1])
def stride(request):
    return request.param


def test_deterministic_waveform_dataset(
    sequential_hanford_background,
    sequential_livingston_background,
    data_size,
    stride,
    sample_rate,
    device,
):
    batch_size = 32
    kernel_length = 1
    dataset = dataloader.DeterministicWaveformDataset(
        sequential_hanford_background,
        sequential_livingston_background,
        kernel_length=kernel_length,
        stride=stride,
        sample_rate=sample_rate,
        batch_size=batch_size,
    )

    assert dataset.waveforms is None
    assert dataset.glitches is None
    assert dataset.background.device.type == "cpu"
    assert dataset.background.dtype == torch.float64

    dataset.to(device)
    assert dataset.background.device.type == device
    assert dataset.background.dtype == torch.float32

    stride_size = int(stride * sample_rate)
    kernel_size = int(kernel_length * sample_rate)

    num_kernels = (data_size - kernel_size) // stride_size + 1
    num_batches, leftover = divmod(num_kernels, batch_size)

    for i, (X, y) in enumerate(dataset):
        assert (y.cpu().numpy() == 0).all()

        X = X.cpu().numpy()
        if i == num_batches and leftover > 0:
            expected_batch = leftover
        else:
            expected_batch = batch_size
        assert X.shape == (expected_batch, 2, sample_rate)

        for j, x in enumerate(X):
            start = stride_size * (i * batch_size + j)
            stop = start + kernel_size
            expected = np.arange(start, stop)

            for k, ifo in enumerate(x):
                power = (-1) ** k
                assert (ifo == power * expected).all()

    assert i == (num_batches if leftover > 0 else num_batches - 1)


def test_deterministic_waveform_dataset_with_glitch_sampling(
    ones_hanford_background,
    ones_livingston_background,
    glitch_sampler,
    glitch_length,
    sample_rate,
    data_length,
    stride,
    device,
):
    batch_size = 8
    kernel_length = 1
    data_size = int(data_length * sample_rate)
    kernel_size = int(kernel_length * sample_rate)
    glitch_size = int(glitch_length * sample_rate)
    stride_size = int(stride * sample_rate)

    if not isinstance(glitch_sampler, (str, Path)):
        glitch_sampler.deterministic = True

    dataset = dataloader.DeterministicWaveformDataset(
        ones_hanford_background,
        ones_livingston_background,
        kernel_length=kernel_length,
        stride=stride,
        sample_rate=sample_rate,
        batch_size=batch_size,
        glitch_sampler=glitch_sampler,
    )
    assert dataset.glitches is not None
    assert dataset.glitches.shape == (10, 2, kernel_size)
    assert dataset.glitches.device.type == "cpu"

    for i, glitch in enumerate(dataset.glitches.numpy()):
        for j, ifo in enumerate(glitch):
            start = glitch_size * i + glitch_size // 2 - kernel_size // 2
            stop = start + kernel_size
            power = (-1) ** j

            assert (ifo == power * np.arange(start, stop)).all()

    dataset.to(device)
    assert dataset.glitches.device.type == device

    num_kernels = (data_size - kernel_size) // stride_size + 1
    num_batches, leftover = divmod(num_kernels, batch_size)
    batches_per_iteration = num_batches if leftover == 0 else (num_batches + 1)

    num_glitches = len(dataset.glitches)
    num_glitch_batches, glitch_leftover = divmod(num_glitches, batch_size // 2)
    if glitch_leftover > 0:
        num_glitch_batches += 1

    for i, (X, y) in enumerate(dataset):
        iteration, idx = divmod(i, batches_per_iteration)
        assert (y.cpu().numpy() == 0).all()

        X = X.cpu().numpy()
        expected_batch = batch_size
        if iteration > 0 and idx == (num_glitch_batches - 1):
            expected_batch = glitch_leftover * 2
        elif iteration == 0 and idx == num_batches and leftover > 0:
            expected_batch = leftover

        assert X.shape == (expected_batch, 2, sample_rate)

        if iteration == 0:
            assert (X == 1).all()
            continue

        for j, x in enumerate(X):
            ifo, glitch_idx = divmod(j, expected_batch // 2)
            glitch_idx += idx * batch_size // 2

            glitch = dataset.glitches[glitch_idx, ifo].cpu().numpy()
            assert (x[ifo] == glitch).all()
            assert (x[1 - ifo] == 1).all()

    assert iteration == 1
    assert idx == (num_glitch_batches - 1)
