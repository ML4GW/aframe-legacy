import pytest
import torch

from bbhnet.data.transforms.standard_scaler import StandardScaler


def test_standard_scaler():
    scaler = StandardScaler(2)
    assert len(list(scaler.parameters())) == 2

    for i, param in enumerate([scaler.mean, scaler.std]):
        assert param.ndim == 1
        assert len(scaler.mean) == 2
        assert (scaler.mean == i).all()

    x = torch.arange(10)
    X = torch.stack([x, x + 1])
    scaler.fit(X)

    assert (scaler.mean == torch.Tensor([45, 55])).all()
    assert (scaler.std == (99 / 12) ** 0.5).all()

    batch = torch.stack([X, X])
    y = scaler(batch)
    assert (y.mean(axis=-1) == 0).all()
    assert (y.std(axis=-1) == 1).all()

    with pytest.raises(ValueError) as exc_info:
        scaler.fit(batch)
    assert str(exc_info.value).startswith("Expected background")
    assert str(exc_info.value).endswith("but found 3")

    for bad_batch in [X, batch[:, :1]]:
        with pytest.raises(ValueError):
            scaler(bad_batch)
