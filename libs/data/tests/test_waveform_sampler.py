import numpy as np
import pytest
from gwpy.frequencseries import FrequencySeries

from bbhnet.data.waveform_sampler import WaveformSampler


@pytest.fixture(params=[10, 20])
def min_snr(request):
    return request.param


@pytest.fixture(params=[20, 40])
def max_snr(request):
    return request.param


@pytest.fixture(params=[["H1"], ["H1", "L1"], ["H1", "L1", "V1"]])
def ifos(request):
    return request.param


def test_waveform_sampler(
    sine_waveforms,
    glitch_length,
    data_length,
    sample_rate,
    min_snr,
    max_snr,
    ifos,
):
    if max_snr >= min_snr:
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
            np.ones(
                10,
            ),
            df=1,
            channel=ifo + ":STRAIN",
        )
        asds.append(fs)
    sampler.fit(1234567890, 1234567990, *asds)
    assert sampler.ifos == ifos
    assert (sampler.background_asd == 1).all()

    snrs = sampler.compute_snrs(sampler.waveforms)
    assert snrs.shape == (10,)
