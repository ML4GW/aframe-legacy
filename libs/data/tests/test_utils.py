import numpy as np
import pytest

from bbhnet.data import utils as data_utils


@pytest.fixture(params=[16, 20, 64])
def size(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def ndim(request):
    return request.param


@pytest.fixture(params=[None, 4, 8, 11])
def N(request):
    return request.param


@pytest.fixture(params=[8, 40, 100])
def trigger_distance_size(request):
    return request.param


def test_sample_kernels(ndim, size, N):
    # set trigger_dist_size = size / 2
    # this replicates behavior
    # that allows t0 to lie anywhere in kernel

    trigger_dist_size = size / 2

    # for 1D arrays we need more data so that we
    # have enough to sample across
    if ndim == 1 and N is None:
        xsize = size
    elif ndim == 1:
        xsize = N * size * 2
    else:
        xsize = 100

    # set up a dummy array for sampling from
    x = np.arange(xsize)
    if ndim > 1:
        x = np.stack([x + i * xsize for i in range(8)])
        if ndim == 3:
            x = x[:, None]
            x = np.concatenate([x, x + xsize], axis=1)

    if ndim == 1 and N is None:
        # must specify number of samples for ndim == 1
        with pytest.raises(ValueError):
            data_utils.sample_kernels(x, size, trigger_dist_size, N)
        return
    elif ndim == 1:
        # timeseries has to be long enough to sample
        # N kernels of size size
        with pytest.raises(ValueError):
            data_utils.sample_kernels(
                x[: N + size - 1], size, trigger_dist_size, N
            )

    # make sure we returned the appropriate number of kernels
    kernels = data_utils.sample_kernels(x, size, trigger_dist_size, N)
    if N is not None:
        assert len(kernels) == N
    else:
        assert len(kernels) == 8

    # make sure that the kernels all have the expected shape
    expected_shape = (size,)
    if ndim == 3:
        expected_shape = (2,) + expected_shape
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
            assert xsize // 2 in kernel % xsize

            # make sure that the kernel is all contiguous ints
            j = kernel[0]
            assert (kernel == np.arange(j, j + size)).all()

            # keep track of which samples the kernels were
            # sampled from to make sure that there's no
            # overlap if N < len(x)
            if N is not None:
                idx_seen.append(j // xsize)
            else:
                assert j // xsize == i

        # verify that there's no overlapping samples
        if N is not None and N <= len(x):
            assert len(idx_seen) == len(list(set(idx_seen)))
    else:
        # similar tests for 3D case, but need to make
        # sure that we have the same slice from each channel
        idx_seen = []
        for i, kernel in enumerate(kernels):
            # verify center of timeseries in kernel
            assert xsize // 2 in kernel[0] % xsize

            # verify contiguous ints and that we have
            # the same slice in each channel
            j = kernel[0, 0]
            expected = np.arange(j, j + size)
            expected = np.stack([expected, expected + xsize])
            assert (kernel == expected).all()

            # keep track of which samples kernels are from
            if N is not None:
                idx_seen.append(j // xsize)
            else:
                assert j // xsize == i

        # verify no overlapping samples
        if N is not None and N <= len(x):
            assert len(idx_seen) == len(list(set(idx_seen)))


def test_sample_kernels_with_trigger_distance(trigger_distance_size, size):

    # test on a lot of samples for robustness
    N = 10000

    # create dummy arrays
    # where the difference between
    # two values in the array
    # is also the amount of samples apart
    xsize = 200
    x = np.arange(xsize)
    x = np.stack([x + i * xsize for i in range(8)])

    kernels = data_utils.sample_kernels(x, size, trigger_distance_size, N)

    for kernel in kernels:

        # the trigger t0 is half way through timeseries
        t0_value = xsize // 2

        # get the center of the sampled kernel
        kernel_center_value = (np.max(kernel) - (size // 2)) % xsize

        # assert that the number of samples between t0 and the center is
        # less than the trigger distance
        assert abs(t0_value - kernel_center_value) <= trigger_distance_size
