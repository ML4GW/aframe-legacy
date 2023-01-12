from unittest.mock import Mock, patch

import numpy as np
import pytest
from datagen.utils.priors import read_priors_from_file
from scipy.stats import ks_2samp


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
    num_events = 1000
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


def test_pdf_from_events(h5py_mock, sample_params):
    prior = read_priors_from_file(h5py_mock)
    interp_sampled = prior.sample(1000)
    for name in sample_params.dtype.names:
        d, p = ks_2samp(interp_sampled[name], sample_params[name])
        assert p < 1e-10
