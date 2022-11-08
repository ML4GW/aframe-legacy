import logging
import shutil
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from generate_background import main as generate_background
from gwpy.timeseries import TimeSeries


@pytest.fixture(scope="function")
def datadir():
    tmpdir = Path(__file__).resolve().parent / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    logging.shutdown()
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="function")
def logdir():
    tmpdir = Path(__file__).resolve().parent / "log"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    logging.shutdown()
    shutil.rmtree(tmpdir)


@pytest.fixture(params=[["H1"], ["H1", "L1"], ["H1", "L1", "V1"]])
def ifos(request):
    return request.param


@pytest.fixture(params=[1024, 2048, 4096])
def sample_rate(request):
    return request.param


@patch("generate_background.query_segments")
def test_generate_background(
    mock_query,
    datadir,
    logdir,
    ifos,
    sample_rate,
):

    start = 1234567890
    stop = 1234577890

    segment_list = [[start, stop]]
    mock_query.return_value = segment_list

    state_flag = "DCS-ANALYSIS_READY_C01:1"
    minimum_length = 10
    channel = "DCS-CALIB_STRAIN_CLEAN_C01"
    frame_type = "HOFT_C01"

    times = np.arange(start, stop, 1 / sample_rate)
    n_samples = len(times)
    ts = TimeSeries(np.ones(n_samples), times=times)

    mock_ts = patch("gwpy.timeseries.TimeSeries.read", return_value=ts)
    mock_datafind = patch("generate_background.find_urls", return_value=None)

    with mock_ts, mock_datafind:
        path = generate_background(
            start,
            stop,
            ifos,
            sample_rate,
            channel,
            frame_type,
            state_flag,
            minimum_length,
            datadir,
            logdir,
        )

    with h5py.File(path) as f:
        print(f.keys())
        print(f.attrs.keys())
        for ifo in ifos:
            assert (f[f"{ifo}:{channel}"] == ts.value).all()
            # assert f["t0"][()] == start
