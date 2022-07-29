from unittest.mock import patch

import numpy as np

from bbhnet.data.glitch_sampler import GlitchSampler


def test_glitch_sampler(
    deterministic,
    arange_glitches,
    glitch_length,
    sample_rate,
    data_length,
    offset,
    device,
    ifos,
    frac,
):
    sampler = GlitchSampler(arange_glitches, deterministic, frac=frac)
    sampler.to(device)

    for ifo in ifos:
        assert sampler.glitches[ifo].device.type == device
        assert sampler.glitches[ifo].device.type == device

    expected_length = glitch_length * sample_rate
    init_idx = 0
    if frac is None:
        expected_num = 100
    else:
        expected_num = abs(frac) * 100
        if frac < 0:
            init_idx = (100 - expected_num) * expected_length

    for ifo in ifos:
        assert sampler.glitches[ifo].shape == (expected_num, expected_length)
        assert sampler.glitches[ifo].shape == (expected_num, expected_length)

    # take over randint to return 4 so that we
    # know what size arrays to expect and verify
    # that they come out correctly
    with patch("numpy.random.randint", return_value=4):
        glitches = sampler.sample(4 * len(ifos), data_length, offset)

    expected_batch = 4 * len(ifos) if deterministic else 4
    for ifo in ifos:
        assert glitches[ifo].shape == (expected_batch, data_length)

    # resample these since setting randint to equal
    # 4 actually throws off sample_kernels when it
    # gets called under the hood. Use this cheap little
    # loop to make sure that we have some data for both
    # interferometers to verify
    for ifo in ifos:
        glitches[ifo] = None
    while any(value is None for value in glitches.values()):
        glitches = sampler.sample(8, data_length, offset)

    glitch_size = glitch_length * sample_rate
    for i, ifo in enumerate(ifos):
        tensor = glitches[ifo]
        value = tensor.cpu().numpy()

        # the livingston data is negative
        power = (-1) ** i

        # make sure each sampled glitch matches our expectations
        for j, row in enumerate(value):
            if deterministic:
                step = glitch_length * sample_rate
                expected = j * step + step // 2 - data_length // 2 + offset
                expected += init_idx
                assert row[0] == power * expected
            else:
                # make sure that the "trigger" of each glitch aka
                # the center value is in each sample
                assert glitch_size // 2 in row % glitch_size

            # make sure that the sampled glitch is a
            # contiguous chunk of ints
            j = row[0]
            assert (row == np.arange(j, j + power * data_length, power)).all()

    if deterministic:
        return

    # now make sure that Nones get returned when
    # all the glitches are one or the other

    n_samples = (len(ifos) - 1) * 4
    with patch("numpy.random.randint", return_value=0):
        glitches = sampler.sample(n_samples, data_length)
        print(glitches)

    for i, tensor in enumerate(glitches.values()):
        # if not on last ifo
        # expect to return None
        if (i + 1) != len(ifos):
            assert tensor is None
        else:
            assert tensor.shape == (n_samples, data_length)

    with patch("numpy.random.randint", return_value=4):
        glitches = sampler.sample(n_samples, data_length)

    for i, tensor in enumerate(glitches.values()):
        # if on last ifo
        # expect to return Non
        if (i + 1) == len(ifos):
            assert tensor is None
        else:
            assert tensor.shape == (4, data_length)
