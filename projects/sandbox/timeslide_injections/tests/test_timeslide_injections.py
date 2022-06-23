import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from timeslide_injections import main


@pytest.fixture(scope="session")
def data_dir():
    data_dir = "tmp"
    os.makedirs(data_dir, exist_ok=True)
    yield Path(data_dir)
    shutil.rmtree(data_dir)


@pytest.fixture(params=[0, 1, 10])
def n_slides(request):
    return request.param


@pytest.fixture(params=[-1, 0, 1])
def shift(request):
    return request.param


@pytest.fixture(params=[512, 1024])
def seg_length(request):
    return request.param


# This test just makes sure that all the expected directories are present
# and that the number of files is consistent between them.
# Nothing within the files is checked.


def test_output_file_structure(data_dir, n_slides, shift, seg_length):

    start = 1258848000
    stop = 1258851000
    prior_file = "priors/nonspin_BBH.prior"
    n_samples = 10
    gw_file = "../O3b_GW_times.txt"

    main(
        start=start,
        stop=stop,
        outdir=data_dir,
        prior_file=prior_file,
        n_samples=n_samples,
        n_slides=n_slides,
        shift=shift,
        seg_length=seg_length,
        gw_file=gw_file,
    )

    dirs = os.listdir(data_dir)

    # Check for all the directories
    assert "original" in dirs
    assert "injected" in dirs
    for ts in np.linspace(0, shift * (n_slides - 1), n_slides):
        assert "dt-{:.1f}".format(ts) in dirs
        dt_dir = os.path.join(data_dir, "dt-{:.1f}".format(ts))
        assert "original" in os.listdir(dt_dir)
        assert "injected" in os.listdir(dt_dir)

    # 1 file per segment for each ifo
    num_segs = len(os.listdir(os.path.join(data_dir, "original"))) / 2
    assert num_segs.is_integer()

    # 3 per segment for injected due to signal parameter file
    assert num_segs * 3 == len(os.listdir(os.path.join(data_dir, "injected")))

    # 1 file per segment per timeslide in both original and injected
    for ts in np.linspace(0, shift * (n_slides - 1), n_slides):
        dt_dir = os.path.join(data_dir, "dt-{:.1f}".format(ts))
        assert num_segs == len(os.listdir(os.path.join(dt_dir, "original")))
        assert num_segs == len(os.listdir(os.path.join(dt_dir, "injected")))
