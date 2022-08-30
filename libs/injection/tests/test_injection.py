#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import bilby
import pytest

import bbhnet.injection

TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture(params=["nonspin_BBH.prior", "precess_tides.prior"])
def prior_file(request):
    return TEST_DIR / "prior_files" / request.param


@pytest.fixture(params=["IMRPhenomPv2"])
def approximant(request):
    return request.param


@pytest.fixture(params=["20", "50"])
def reference_frequency(request):
    return request.param


@pytest.fixture(params=[10, 20])
def minimum_frequency(request):
    return request.param


@pytest.fixture(params=[1, 100])
def n_samples(request):
    return request.param


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[2, 4, 8])
def waveform_duration(request):
    return request.param


def test_generate_gw(
    prior_file,
    approximant,
    sample_rate,
    waveform_duration,
    minimum_frequency,
    reference_frequency,
    n_samples,
):

    n_pols = 2
    prior = bilby.gw.prior.PriorDict(str(prior_file))
    sample_params = prior.sample(n_samples)

    waveforms = bbhnet.injection.injection.generate_gw(
        sample_params,
        minimum_frequency,
        reference_frequency,
        sample_rate,
        waveform_duration,
        approximant,
    )

    expected_waveform_size = waveform_duration * sample_rate
    expected_num_waveforms = n_samples

    assert waveforms.shape == (
        expected_num_waveforms,
        n_pols,
        expected_waveform_size,
    )
