import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    return request.param


@pytest.fixture(scope="function")
def data_dir():
    data_dir = "tmp"
    os.makedirs(data_dir, exist_ok=True)
    yield Path(data_dir)
    shutil.rmtree(data_dir)


@pytest.fixture(params=[512, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[2, 4])
def glitch_length(request):
    return request.param


@pytest.fixture
def write_timeseries(data_dir):
    def func(fname, **kwargs):
        with h5py.File(data_dir / fname, "w") as f:
            for key, value in kwargs:
                f[key] = value
        return data_dir / fname

    return func


@pytest.fixture
def arange_glitches(glitch_length, sample_rate, write_timeseries):
    data = {
        "H1_glitches": np.arange(glitch_length * sample_rate),
        "L1_glitches": -np.arange(glitch_length * sample_rate),
    }
    write_timeseries("arange_glitches.h5", **data)
    return "arange_glitches.h5"
