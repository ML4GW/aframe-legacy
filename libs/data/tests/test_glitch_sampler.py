from unittest.mock import patch

import numpy as np

from bbhnet.data.glitch_sampler import GlitchSampler


def test_glitch_sampler(
    arange_glitches, glitch_length, sample_rate, data_length, device
):
    sampler = GlitchSampler(arange_glitches, device)
    assert len(sampler.hanford) == glitch_length * sample_rate
    assert len(sampler.livingston) == glitch_length * sample_rate

    with patch("numpy.random.randint", return_value=4):
        hanford, livingston = sampler.sample(8, data_length)
    assert hanford.shape == (4, data_length)
    assert livingston.shape == (4, data_length)

    for tensor in [hanford, livingston]:
        value = tensor.cpu().numpy()
        assert (glitch_length * sample_rate) // 2 in value

        i = value[0]
        assert (value == np.arange(i, i + data_length)).all()

    with patch("numpy.random.randint", return_value=0):
        hanford, livingston = sampler.sample(8, data_length)
    assert hanford is None
    assert livingston.shape == (8, data_length)

    with patch("numpy.random.randint", return_value=8):
        hanford, livingston = sampler.sample(8, data_length)
    assert livingston is None
    assert hanford.shape == (8, data_length)
