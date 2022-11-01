from unittest.mock import MagicMock, patch

import torch

from bbhnet.data.waveform_injection import BBHNetWaveformInjection


def sample(N):
    return torch.ones((N, 2, 128 * 2))


rand_value = 0.1 + 0.5 * torch.arange(32) % 2


@patch("ml4gw.transforms.injection.RandomWaveformInjection.sample", new=sample)
@patch("torch.rand", return_value=rand_value)
def test_bbhnet_waveform_injection(rand_mock):
    tform = BBHNetWaveformInjection(
        sample_rate=128,
        ifos=["H1", "L1"],
        dec=MagicMock(),
        psi=MagicMock(),
        phi=MagicMock(),
        prob=0.5,
        plus=MagicMock(),
        cross=MagicMock(),
    )

    X = torch.zeros((32, 2, 128 * 1))
    y = torch.zeros((32, 1))

    X, y = tform(X, y)
    assert (X[::2] == 1).all().item()
    assert (y[::2] == 1).all().item()
