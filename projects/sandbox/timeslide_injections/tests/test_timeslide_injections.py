import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries
from timeslide_injections import main

from bbhnet.io.timeslides import TimeSlide


@pytest.fixture(scope="session")
def outdir():

    outdir = Path(__file__).resolve().parent / "tmp"
    outdir.mkdir(exist_ok=False, parents=True)
    yield outdir
    shutil.rmtree(outdir)


@pytest.fixture(params=["priors/nonspin_BBH.prior"])
def prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


@pytest.fixture(params=[60])
def spacing(request):
    return request.param


@pytest.fixture(params=[8])
def buffer(request):
    return request.param


@pytest.fixture(params=[1, 2])
def n_slides(request):
    return request.param


@pytest.fixture(params=[4096])
def file_length(request):
    return request.param


@pytest.fixture(params=[32])
def fmin(request):
    return request.param


@pytest.fixture(params=[2048])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[[0, 1]])
def shifts(request):
    return request.param


@pytest.fixture(params=[["H1", "L1"]])
def ifos(request):
    return request.param


@pytest.fixture(params=["HOFT_C01"])
def frame_type(request):
    return request.param


@pytest.fixture(params=["DCS-CALIB_STRAIN_CLEAN_C01"])
def channel(request):
    return request.param


def test_timeslide_injections(
    outdir,
    prior_file,
    spacing,
    buffer,
    n_slides,
    shifts,
    file_length,
    ifos,
    fmin,
    sample_rate,
    frame_type,
    channel,
):

    start = 1123456789
    stop = 1123457789

    times = np.arange(start, stop, 1 / sample_rate)
    n_samples = len(times)
    ts = TimeSeries(np.ones(n_samples), times=times)

    mock_ts = patch("gwpy.timeseries.TimeSeries.read", return_value=ts)

    mock_datafind = patch("gwdatafind.find_urls", return_value=None)

    with mock_datafind, mock_ts:
        outdir = main(
            start,
            stop,
            outdir,
            prior_file,
            spacing,
            buffer,
            n_slides,
            shifts,
            ifos,
            file_length,
            fmin,
            sample_rate,
            frame_type,
            channel,
        )

    timeslides = outdir.iterdir()
    timeslides = [slide for slide in timeslides if slide.is_dir()]
    timeslides = list(timeslides)
    assert len(timeslides) == n_slides

    for slide in timeslides:
        injection_ts = TimeSlide(slide, field="injection")
        background_ts = TimeSlide(slide, field="background")

        assert len(injection_ts.segments) == len(background_ts.segments)
