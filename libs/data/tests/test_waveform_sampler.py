from unittest.mock import patch

import numpy as np
import pytest
from gwpy.frequencyseries import FrequencySeries

from bbhnet.data.waveform_sampler import WaveformSampler
from bbhnet.injection.injection import calc_snr


@pytest.fixture(params=[10, 20])
def min_snr(request):
    return request.param


@pytest.fixture(params=[20, 40])
def max_snr(request):
    return request.param


@pytest.fixture(params=[["H1"], ["H1", "L1"], ["H1", "L1", "V1"]])
def ifos(request):
    return request.param


# def project_raw_gw(waveform, sample_params):

#     return waveform


def test_waveform_sampler(
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

    sampler = WaveformSampler(sine_waveforms, sample_rate, min_snr, max_snr)
    assert sampler.waveforms.shape == (10, glitch_length * sample_rate)
    with pytest.raises(RuntimeError):
        sampler.sample(8, data_length)

    asds = []
    for ifo in ifos:
        fs = FrequencySeries(
            np.ones((sample_rate // 2,)),
            df=2 / sample_rate + 1,
            channel=ifo + ":STRAIN",
        )
        asds.append(fs)
    sampler.fit(1234567890, 1234567990, *asds)
    assert sampler.ifos == ifos
    assert (sampler.background_asd == 1).all()

    waveforms = sampler.waveforms[:, None]
    multichannel = np.concatenate(
        [waveforms * 0.5**i for i in range(len(ifos))], axis=1
    )

    snrs = sampler.compute_snrs(multichannel)
    assert snrs.shape == (10, len(ifos))
    for row, sample in zip(snrs, multichannel):
        for snr, ifo in zip(row, sample):
            assert np.isclose(snr, calc_snr(ifo, fs, sample_rate), rtol=1e-9)

    target_snrs = np.arange(1, 11)
    with patch("numpy.random.uniform", return_value=target_snrs):
        reweighted = sampler.reweight_snrs(multichannel)

    for target, sample in zip(target_snrs, reweighted):
        calcd = 0
        for ifo in zip(row, sample):
            calcd += calc_snr(ifo, fs, sample_rate) ** 2
        assert np.isclose(calcd**0.5, target, rtol=1e-9)

    # TODO: do this again with project_raw_gw patched
    # to just return the waveform as-is and verify the
    # sample params. Also patch reweight_snrs to return
    # as-is so we can do some check to make sure the
    # "trigger" is in there. Should we apply some sort
    # of gaussian to the waves so that there's a unique
    # max value we can check for?
    results = sampler.sample(4, data_length)
    assert results.shape == (4, len(ifos), data_length)
    for sample in results:
        calcd = 0
        for ifo in sample:
            calcd += calc_snr(ifo, fs, sample_rate) ** 2
        assert min_snr < calcd**0.5 < max_snr
