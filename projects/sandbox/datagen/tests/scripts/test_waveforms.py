from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
from datagen.scripts import generate_waveforms
from datagen.utils.priors import (
    end_o3_ratesandpops,
    extrinsic_params,
    nonspin_bbh,
)


@pytest.fixture(params=[0, 10, 100])
def n_samples(request):
    return request.param


@pytest.fixture(params=[50])
def reference_frequency(request):
    return request.param


@pytest.fixture(params=[20, 40])
def minimum_frequency(request):
    return request.param


@pytest.fixture(params=[8])
def waveform_duration(request):
    return request.param


@pytest.fixture(params=[512, 4096, 16384])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[extrinsic_params, nonspin_bbh, end_o3_ratesandpops])
def prior(request):
    return request.param


@pytest.fixture
def sample_params():
    event_dtype = np.dtype(
        [
            ("mass_1", "<f8"),
            ("mass_ratio", "<f8"),
            ("a_1", "<f8"),
            ("a_2", "<f8"),
            ("cos_tilt_1", "<f8"),
            ("cos_tilt_2", "<f8"),
            ("redshift", "<f8"),
            ("mass_2", "<f8"),
        ]
    )
    num_events = 19
    rand = np.random.rand(num_events)
    mass_1 = rand * 100
    mass_ratio = rand
    a_1 = rand
    a_2 = rand
    cos_tilt_1 = rand * 2 - 1
    cos_tilt_2 = rand * 2 - 1
    redshift = rand * 1.5 + 0.05
    mass_2 = np.multiply(mass_1, mass_ratio)
    # There must be a better way, but this works
    params = np.array(
        [
            (m1, mr, a1, a2, ct1, ct2, z, m2)
            for m1, mr, a1, a2, ct1, ct2, z, m2 in zip(
                mass_1,
                mass_ratio,
                a_1,
                a_2,
                cos_tilt_1,
                cos_tilt_2,
                redshift,
                mass_2,
            )
        ],
        dtype=event_dtype,
    )

    return params


@pytest.fixture
def h5py_mock(sample_params):
    def mock(fname, _):
        value = {"events": sample_params}
        obj = Mock()
        obj.__enter__ = lambda obj: value
        obj.__exit__ = Mock()
        return obj

    with patch("h5py.File", new=mock):
        yield mock


def test_check_file_contents(
    datadir,
    logdir,
    h5py_mock,
    n_samples,
    waveform_duration,
    sample_rate,
    prior,
    minimum_frequency,
    reference_frequency,
):
    signal_file = generate_waveforms(
        prior,
        n_samples,
        logdir,
        datadir,
        reference_frequency,
        minimum_frequency,
        sample_rate,
        waveform_duration,
        parameter_file="param_file.h5",
    )

    with h5py.File(signal_file, "r") as f:
        for key in f.keys():
            if key == "signals":
                act_shape = f[key].shape
                exp_shape = (n_samples, 2, waveform_duration * sample_rate)
                assert (
                    act_shape == exp_shape
                ), f"Expected shape {exp_shape} for signals, found {act_shape}"
            else:
                act_shape = f[key].shape
                exp_shape = (n_samples,)
                assert (
                    act_shape == exp_shape
                ), f"Expected shape {exp_shape} for {key}, found {act_shape}"
