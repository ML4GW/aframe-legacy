import numpy as np
import pytest

from bbhnet.data import utils as data_utils


@pytest.fixture(params=[128, 300, 512])
def size(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def ndim(request):
    return request.param


@pytest.fixture(params=[None, 4, 8, 11])
def N(request):
    return request.param


def test_sample_kernels(ndim, size, N):
    x = np.arange(100)
    if ndim > 1:
        x = np.stack([x + 100 for i in range(8)])
        if ndim == 3:
            x = x[:, None]
            x = np.concatenate([x, x + 100], axis=1)

    if ndim == 1 and N is None:
        # must specify number of samples for ndim == 1
        with pytest.raises(ValueError):
            data_utils.sample_kernels(x, size, N)
        return
    elif ndim == 1:
        # timeseries has to be long enough to sample
        # N kernels of size size
        with pytest.raises(ValueError):
            data_utils.sample_kernels(x[: N + size - 1], size, N)

    # make sure we returned the appropriate number of kernels
    kernels = data_utils.sample_kernels(x, size, N)
    if N is not None:
        assert len(kernels) == N
    else:
        assert len(kernels) == 8

    # make sure that the kernels all have the expected shape
    expected_shape = (size,)
    if ndim == 2:
        expected_shape = (8,) + expected_shape
    elif ndim == 3:
        expected_shape = (8, 2) + expected_shape
    assert all([i.shape == expected_shape for i in kernels])

    # verify kernel content
    if ndim == 1:
        # 1D case is easy
        for kernel in kernels:
            i = kernel[0]
            assert (kernel == np.arange(i, i + size)).all()
    elif ndim == 2:
        # 2D needs to check more
        idx_seen = []
        for i, kernel in enumerate(kernels):
            # make sure the center of the timeseries is in kernel
            assert 50 in kernel % 100

            # make sure that the kernel is all contiguous ints
            j = kernel[0]
            assert (kernel == np.arange(j, j + size)).all()

            # keep track of which samples the kernels were
            # sampled from to make sure that there's no
            # overlap if N < len(x)
            idx_seen.append(j // 100)

        # verify that there's no overlapping samples
        if N <= len(x):
            assert len(idx_seen) == len(list(set(idx_seen)))
    else:
        # similar tests for 3D case, but need to make
        # sure that we have the same slice from each channel
        idx_seen = []
        for i, kernel in enumerate(kernels):
            # verify center of timeseries in kernel
            assert 50 in kernel[0] % 100

            # verify contiguous ints and that we have
            # the same slice in each channel
            j = kernel[0, 0]
            expected = np.arange(j, j + size)
            expected = np.stack([expected, expected + 100])
            assert (kernel == expected).all()

            # keep track of which samples kernels are from
            idx_seen.append(j // 100)

        # verify no overlapping samples
        if N <= len(x):
            assert len(idx_seen) == len(list(set(idx_seen)))
