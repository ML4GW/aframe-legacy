import pytest
import torch

from bbhnet.io.transforms.transform import Transform, _make_tensor


@pytest.fixture(params=[torch.int64, torch.float32])
def tensor_type(request):
    return request.param


def test_make_tensor(tensor_type, device):
    # test default behavior
    t = _make_tensor(1.0, device)
    assert isinstance(t, torch.Tensor)
    assert t.ndim == 0
    assert t.type() == torch.float32
    assert t.device == device

    t = _make_tensor([0.0, 1.0], device)
    assert isinstance(t, torch.Tensor)
    assert t.ndim == 1
    assert t.type() == torch.float32
    assert t.device == device

    # now run same tests with explicit tensor type
    t = _make_tensor(1.0, device, tensor_type)
    assert isinstance(t, torch.Tensor)
    assert t.ndim == 0
    assert t.type() == tensor_type
    assert t.device == device

    t = _make_tensor([0.0, 1.0], device, tensor_type)
    assert isinstance(t, torch.Tensor)
    assert t.ndim == 1
    assert t.type() == tensor_type
    assert t.device == device


def test_transform(tensor_type, device):
    # test `add_parameter` for scalar and tensor
    param = Transform.add_parameter(None, 1.0, device, tensor_type)
    assert isinstance(param, torch.nn.Parameter)
    assert not param.requires_grad
    assert param.value == 1.0
    assert param.ndim == 0
    assert param.type() == tensor_type
    assert param.device == device

    param = Transform.add_parameter(None, [0.0, 1.0], device, tensor_type)
    assert isinstance(param, torch.nn.Parameter)
    assert not param.requires_grad
    assert all([i == j for i, j in zip(param.value, range(2))])
    assert param.ndim == 1
    assert param.type() == tensor_type
    assert param.device == device

    # test `set_value`
    Transform.set_value(None, param, [1.0, 2.0])
    assert not param.requires_grad
    assert all([i == j for i, j in zip(param.value, range(1, 3))])
    assert param.ndim == 1
    assert param.type() == tensor_type
    assert param.device == device

    # this will raise an error, need to figure out what
    Transform.set_value(None, param, 1.0)
