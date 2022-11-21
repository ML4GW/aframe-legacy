import logging

import pytest


@pytest.fixture(scope="function")
def datadir(tmp_path):
    datadir = tmp_path.mkdir(parents=True, exist_ok=False) / "data"
    return datadir


@pytest.fixture(scope="function")
def logdir(tmp_path):
    logdir = tmp_path.mkdir(parents=True, exist_ok=False) / "log"
    yield logdir
    logging.shutdown()
