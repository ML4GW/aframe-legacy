from unittest.mock import patch

import numpy as np

from bbhnet.data.glitch_sampler import GlitchSampler


def test_glitch_sampler(
    arange_glitches, glitch_length, sample_rate, data_length, device
):
    sampler = GlitchSampler(arange_glitches, device)
    assert sampler.hanford.shape == (10, glitch_length * sample_rate)
    assert sampler.livingston.shape == (10, glitch_length * sample_rate)

    with patch("numpy.random.randint", return_value=4):
        hanford, livingston = sampler.sample(8, data_length)
    assert hanford.shape == (4, data_length)
    assert livingston.shape == (4, data_length)

    hanford, livingston = sampler.sample(8, data_length)
    glitch_size = glitch_length * sample_rate
    for i, tensor in enumerate([hanford, livingston]):
        value = tensor.cpu().numpy()
        power = (-1)**i
        for row in value:
            j = row[0]
            assert glitch_size // 2 in row % glitch_size
            assert (row == np.arange(j, j + power * data_length, power)).all()

    with patch("numpy.random.randint", return_value=0):
        hanford, livingston = sampler.sample(8, data_length)
    assert hanford is None
    assert livingston.shape == (8, data_length)

    with patch("numpy.random.randint", return_value=8):
        hanford, livingston = sampler.sample(8, data_length)
    assert livingston is None
    assert hanford.shape == (8, data_length)
