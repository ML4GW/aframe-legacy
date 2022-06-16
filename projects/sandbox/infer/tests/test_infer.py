from unittest.mock import patch

import pytest
from infer import main as infer


@pytest.fixture
def tmpdir(tmp_path):
    tmp_path.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def data_dir(tmpdir):
    data_dir = tmpdir / "data"
    data_dir.mkdir()
    return data_dir


# TODO: how to patch everything around Triton
# to make inference requests just add 1 to the
# last element of each stream?
@patch("tritonserve.serve")
def test_infer():
    infer()
