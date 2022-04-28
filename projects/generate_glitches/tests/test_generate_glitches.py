#!/usr/bin/env python
# coding: utf-8
import os
import shutil
from pathlib import Path

import pytest
from generate_glitches import generate_glitch_dataset

TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def data_dir():
    data_dir = "tmp"
    os.makedirs(data_dir, exist_ok=True)
    yield Path(data_dir)
    shutil.rmtree(data_dir)


@pytest.fixture(params=[2, 4])
def window(request):
    return request.param


@pytest.fixture(params=["HOFT_C01"])
def frame_type(request):
    return request.param


@pytest.fixture(params=["DCS-CALIB_STRAIN_CLEAN_C01"])
def channel(request):
    return request.param


@pytest.fixture(params=[9, 11])
def snr_thresh(request):
    return request.param


@pytest.fixture(params=["H1", "L1"])
def ifo(request):
    return request.param


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture()
def omicron_dir(request):
    return "/home/ethan.marx/bbhnet/generate-glitch-dataset/omicron/"


@pytest.fixture()
def trig_file(ifo):
    return str(TEST_DIR / "triggers" / f"triggers_{ifo}.txt")


def test_glitch_data_shape_and_glitch_snrs(
    data_dir,
    ifo,
    window,
    sample_rate,
    snr_thresh,
    trig_file,
    channel,
    frame_type,
):
    start = 1263588390
    stop = 1263592390

    glitch_len = 2 * window * sample_rate

    glitches, snrs = generate_glitch_dataset(
        ifo,
        snr_thresh,
        start,
        stop,
        window,
        sample_rate,
        channel,
        frame_type,
        trig_file,
    )

    assert glitches.shape[-1] == glitch_len
    assert len(glitches) == len(snrs)
    assert all(snrs > snr_thresh)
