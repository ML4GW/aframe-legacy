import os
import shutil
import sys

import numpy as np
import pytest

from bbhnet.trainer.wrapper import trainify


@pytest.fixture
def output_directory():
    os.makedirs("tmp")
    yield "tmp"
    shutil.rmtree("tmp")


def make_random_data(
    length: int, output_directory: str, verbose: bool = False, **kwargs
):
    return np.random.randn(10, length), np.random.randn(length)


def test_wrapper(output_directory):
    fn = trainify(make_random_data)

    # make sure we can run the function as-is with regular arguments
    X, y = fn(100, output_directory)
    assert X.shape == (10, 100)
    assert y.shape == (100,)

    # call function passing keyword args
    # for train function
    result = fn(
        4096,
        output_directory,
        kernel_length=1,
        kernel_stride=128 / 4096,
        sample_rate=256,
        max_epochs=1,
        arch="autoencoder",
    )
    assert len(result["train_loss"]) == 1

    sys.argv = [
        None,
        "--length",
        "4096",
        "--output-directory",
        output_directory,
        "--kernel-length",
        "1",
        "--kernel-stride",
        str(128 / 4096),
        "--sample-rate",
        "256",
        "--max-epochs",
        "1",
        "--chunk-length",
        "0",
        "--alpha",
        "0",
        "autoencoder",
    ]

    # since trainify wraps function w/ typeo
    # looks for args from command line
    # i.e. from sys.argv
    result = fn()
    assert len(result["train_loss"]) == 1
