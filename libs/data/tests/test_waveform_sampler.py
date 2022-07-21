from unittest.mock import patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from bbhnet.data.waveform_sampler import WaveformSampler
from bbhnet.injection.injection import calc_snr


@pytest.fixture(params=[10, 20])
def min_snr(request):
    return request.param


@pytest.fixture(params=[20, 40])
def max_snr(request):
    return request.param


@pytest.fixture(params=[True, False])
def deterministic(request):
    return request.param


def test_waveform_sampler(
    deterministic,
    sine_waveforms,
    glitch_length,
    data_length,
    sample_rate,
    min_snr,
    max_snr,
    ifos,
):
    if max_snr <= min_snr:
        with pytest.raises(ValueError):
            WaveformSampler(sine_waveforms, sample_rate, min_snr, max_snr)
        return

    sampler = WaveformSampler(
        sine_waveforms,
        sample_rate,
        min_snr,
        max_snr,
        deterministic=deterministic,
    )
    assert sampler.waveforms.shape == (10, 2, glitch_length * sample_rate)

    # we haven't fit to a background yet, so trying to sample
    # should raise an error because we can't do snr refitting
    with pytest.raises(RuntimeError):
        sampler.sample(8, data_length)

    # fit the sampler to these backgrounds and make sure
    # that the saved attributes are correct
    # TODO: how can we make waveforms that have a known
    # ASD that won't have 0s? Should we patch here?
    background = np.random.randn(sample_rate * 100)
    args = {ifo: background for ifo in ifos}
    sampler.fit(**args)
    assert sampler.ifos == ifos
    assert sampler.background_asd.shape[-1] == (sample_rate * 100 // 2 + 1)

    # create an array with a dummy 1st dimension to
    # replicate a waveform projected onto multiple ifos
    waveforms = sampler.waveforms[:, :1]
    multichannel = np.concatenate(
        [waveforms * 0.5**i for i in range(len(ifos))], axis=1
    )

    ts = TimeSeries(background, dt=1 / 4096)
    fs = ts.asd(fftlength=2, window="hanning", method="median")

    # use this to verify that the array-wise computed
    # snrs match the values from calc_snr
    snrs = sampler.compute_snrs(multichannel)
    assert snrs.shape == (10, len(ifos))
    for row, sample in zip(snrs, multichannel):
        for snr, ifo in zip(row, sample):
            assert np.isclose(snr, calc_snr(ifo, fs, sample_rate), rtol=1e-9)

    # patch numpy.random.uniform so that we know which
    # reweighted snr values these waveforms will be
    # mapped to after reweighting, use it to verify the
    # functionality of reweight_snrs
    target_snrs = np.arange(1, 11)
    with patch("numpy.random.uniform", return_value=target_snrs):
        reweighted = sampler.reweight_snrs(multichannel)

    if deterministic:
        target_snrs = np.ones((10,)) * (min_snr * max_snr) ** 0.5

    for target, sample in zip(target_snrs, reweighted):
        calcd = 0
        for ifo in sample:
            calcd += calc_snr(ifo, fs, sample_rate) ** 2
        assert np.isclose(calcd**0.5, target, rtol=1e-9)

    # TODO: do this again with project_raw_gw patched
    # to just return the waveform as-is and verify the
    # sample params. Also patch reweight_snrs to return
    # as-is so we can do some check to make sure the
    # "trigger" is in there. Should we apply some sort
    # of gaussian to the waves so that there's a unique
    # max value we can check for?
    # TODO: check to make "trigger" is in middle for deterministic case
    results = sampler.sample(4, data_length, 100)
    assert len(results) == 4
    assert all([i.shape == (len(ifos), data_length) for i in results])

    # now sample using the entire waveforms to check
    # the SNR ranges. There's definitely a better,
    # more explicit check to do here with patching
    # but this will work for now.
    results = sampler.sample(4, sampler.waveforms.shape[-1], 100)
    for sample in results:
        calcd = 0
        for ifo in sample:
            calcd += calc_snr(ifo, fs, sample_rate) ** 2

        if deterministic:
            assert np.isclose(calcd, (min_snr * max_snr) ** 0.5, rtol=1e-9)
        else:
            assert min_snr < calcd**0.5 < max_snr

    # build "backgroud" asds of all 0s
    # to test that exception is raised
    args = {ifo: np.zeros((sample_rate * 100,)) for ifo in ifos}
    with pytest.raises(ValueError):
        sampler.fit(**args)
