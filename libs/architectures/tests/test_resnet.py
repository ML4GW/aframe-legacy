import pytest
import torch

from bbhnet.architectures.resnet import BasicBlock, Bottleneck, ResNet, conv1


@pytest.fixture(params=[3, 7, 8, 11])
def kernel_size(request):
    return request.param


@pytest.fixture(params=[1024, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[1, 2])
def stride(request):
    return request.param


@pytest.fixture(params=[2, 4])
def inplanes(request):
    return request.param


def test_basic_block(kernel_size, stride, sample_rate, inplanes):
    planes = 4

    if stride > 1 or inplanes != planes:
        downsample = conv1(inplanes, planes, stride)
    else:
        downsample = None

    if kernel_size % 2 == 0:
        with pytest.raises(ValueError):
            block = BasicBlock(
                inplanes, planes, kernel_size, stride, downsample=downsample
            )
        return

    block = BasicBlock(
        inplanes, planes, kernel_size, stride, downsample=downsample
    )
    x = torch.randn(8, inplanes, sample_rate)
    y = block(x)

    assert len(y.shape) == 3
    assert y.shape[1] == 4
    assert y.shape[2] == sample_rate // stride


def test_bottleneck_block(kernel_size, stride, sample_rate, inplanes):
    # TODO: fill out
    if kernel_size % 2 == 0:
        with pytest.raises(ValueError):
            Bottleneck(inplanes, 4, kernel_size, stride)
        return

    Bottleneck(inplanes, 4, kernel_size, stride)
    return


@pytest.fixture(params=[1, 2, 3])
def num_ifos(request):
    return request.param


@pytest.fixture(params=[[2, 2, 2, 2], [2, 4, 4], [2, 2, 4, 2]])
def layers(request):
    return request.param


@pytest.fixture(params=[None, "stride", "dilation"])
def stride_type(request):
    return request.param


def test_resnet(kernel_size, layers, num_ifos, sample_rate, stride_type):
    if kernel_size % 2 == 0:
        with pytest.raises(ValueError):
            nn = ResNet(num_ifos, layers, kernel_size)
        return

    if stride_type is not None:
        stride_type = [stride_type] * (len(layers) - 1)

    if stride_type is not None and stride_type[0] == "dilation":
        with pytest.raises(NotImplementedError):
            nn = ResNet(num_ifos, layers, kernel_size, stride_type=stride_type)
        return

    nn = ResNet(num_ifos, layers, kernel_size, stride_type=stride_type)
    x = torch.randn(8, num_ifos, sample_rate)
    y = nn(x)
    assert y.shape == (8, 1)

    with pytest.raises(ValueError):
        stride_type = ["stride"] * len(layers)
        nn = ResNet(num_ifos, layers, kernel_size, stride_type=stride_type)
    with pytest.raises(ValueError):
        stride_type = ["strife"] * (len(layers) - 1)
        nn = ResNet(num_ifos, layers, kernel_size, stride_type=stride_type)
