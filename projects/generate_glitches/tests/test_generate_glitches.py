#!/usr/bin/env python
# coding: utf-8
import os
import shutil
from pathlib import Path

import pytest
from generate_glitches import generate_glitch_dataset


@pytest.fixture(scope="session")
def data_dir():
    data_dir = "tmp"
    os.makedirs(data_dir, exist_ok=True)
    yield Path(data_dir)
    shutil.rmtree(data_dir)


@pytest.fixture(params=[2, 4])
def window(request):
    return request.param


@pytest.fixture(params=[8])
def snr_thresh(request):
    return request.param


@pytest.fixture(params=["H1"])
def ifo(request):
    return request.param


@pytest.fixture(params=[1024, 2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture()
def omicron_dir(request):
    return "/home/ethan.marx/bbhnet/generate-glitch-dataset/omicron/"


@pytest.fixture(params=[1263565618])
def start(request):
    return request.param


@pytest.fixture(params=[1263569618])
def stop(request):
    return request.param


@pytest.fixture(params=["./triggers/triggers_H1.txt"])
def trig_file(request):
    return request.param


def test_glitch_data_shape(
    data_dir,
    ifo,
    window,
    sample_rate,
    snr_thresh,
    start,
    stop,
    trig_file,
):

    glitch_len = 2 * window * sample_rate

    glitches, snrs = generate_glitch_dataset(
        ifo, snr_thresh, start, stop, window, sample_rate, trig_file
    )

    assert glitches.shape[-1] == glitch_len
    assert len(glitches) == len(snrs)


def test_glitch_snrs(
    data_dir,
    ifo,
    window,
    sample_rate,
    snr_thresh,
    start,
    stop,
    trig_file,
):

    glitches, snrs = generate_glitch_dataset(
        ifo, snr_thresh, start, stop, window, sample_rate, trig_file
    )

    assert all(snrs > snr_thresh)
