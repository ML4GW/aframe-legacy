from unittest.mock import patch

import pytest
import torch

from bbhnet.data.glitch_sampler import GlitchSampler


@pytest.fixture(params=[0, 0.5, 1])
def prob(request):
    return request.param


def get_rand_patch(size):
    probs = torch.linspace(0, 1, size[1])
    return torch.stack([probs] * size[0])


def test_glitch_sampler(
    arange_glitches,
    glitch_length,
    sample_rate,
    data_length,
    offset,
    device,
    prob,
):
    sampler = GlitchSampler(
        prob=prob,
        max_offset=offset,
        h1=arange_glitches[0],
        l1=arange_glitches[1],
    )
    sampler.to(device)
    for glitch in sampler.glitches:
        assert glitch.device.type == device

    X = torch.zeros((8, 2, 512))
    probs = get_rand_patch((2, 8))
    with patch("torch.rand", return_value=probs):
        inserted, _ = sampler(X, None)

        # TODO: the tests could be more extensive, but
        # then are we functionally just testing sample_kernels?
        if prob == 0:
            assert (inserted == 0).all().item()
        elif prob == 1:
            assert (inserted != 0).all().item()
        else:
            assert (inserted[:4] == 0).all().item()
            assert (inserted[4:] != 0).all().item()
