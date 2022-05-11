import os
import sys

import h5py
import numpy as np
import pytest

from bbhnet.trainer.wrapper import trainify


@pytest.fixture(scope="session")
def output_directory(tmpdir_factory):
    out_dir = tmpdir_factory.mktemp("out")
    return out_dir


@pytest.fixture(scope="session")
def data_directory(tmpdir_factory):
    datadir = tmpdir_factory.mktemp("data")
    create_all_data_files(datadir)
    return datadir


def create_data_file(data_dir, filename, dataset_names, data):
    with h5py.File(data_dir.join(filename), "w") as f:
        for name, data in zip(dataset_names, data):
            f.create_dataset(name, data=data)


def create_all_data_files(datadir):

    sample_rate = 2048
    waveform_duration = 4
    signal_length = waveform_duration * sample_rate
    fake_background = np.random.randn(1000 * signal_length)
    fake_glitches = np.random.randn(100, signal_length)
    fake_waveforms = np.random.randn(100, 2, signal_length)

    create_data_file(
        datadir,
        "hanford_background.h5",
        ["hoft", "t0"],
        [fake_background, 1001],
    )
    create_data_file(
        datadir,
        "livingston_background.h5",
        ["hoft", "t0"],
        [fake_background, 1001],
    )

    create_data_file(
        datadir,
        "hanford_background_val.h5",
        ["hoft", "t0"],
        [fake_background, 1001],
    )
    create_data_file(
        datadir,
        "livingston_background_val.h5",
        ["hoft", "t0"],
        [fake_background, 1001],
    )

    create_data_file(
        datadir,
        "glitches.h5",
        ["H1_glitches", "L1_glitches"],
        [fake_glitches, fake_glitches],
    )
    create_data_file(
        datadir,
        "glitches_val.h5",
        ["H1_glitches", "L1_glitches"],
        [fake_glitches, fake_glitches],
    )

    create_data_file(datadir, "signals.h5", ["signals"], [fake_waveforms])
    create_data_file(datadir, "signals_val.h5", ["signals"], [fake_waveforms])


def return_random_data_files(
    data_directory: str, output_directory: str, **kwargs
):

    train_files = {
        "glitch dataset": os.path.join(data_directory, "glitches.h5"),
        "signal dataset": os.path.join(data_directory, "signals.h5"),
        "hanford background": os.path.join(
            data_directory, "hanford_background.h5"
        ),
        "livingston background": os.path.join(
            data_directory, "livingston_background.h5"
        ),
    }

    val_files = {
        "glitch dataset": os.path.join(data_directory, "glitches_val.h5"),
        "signal dataset": os.path.join(data_directory, "signals.h5"),
        "hanford background": os.path.join(
            data_directory, "hanford_background_val.h5"
        ),
        "livingston background": os.path.join(
            data_directory, "livingston_background_val.h5"
        ),
    }

    return train_files, val_files


def test_wrapper(data_directory, output_directory):

    fn = trainify(return_random_data_files)

    # make sure we can run the function as-is with regular arguments
    train_files, val_files = fn(data_directory, output_directory)
    assert "glitch dataset" in train_files.keys()
    assert "hanford background" in val_files.keys()

    # call function passing keyword args
    # for train function
    result = fn(
        data_directory,
        output_directory,
        waveform_frac=0.2,
        glitch_frac=0.3,
        kernel_length=1,
        batch_size=10,
        batches_per_epoch=2,
        sample_rate=2048,
        max_epochs=1,
        arch="resnet",
        layers=[2, 2, 2],
    )
    assert len(result["train_loss"]) == 1

    sys.argv = [
        None,
        "--data-directory",
        str(data_directory),
        "--output-directory",
        str(output_directory),
        "--kernel-length",
        "1",
        "--waveform-frac",
        "0.3",
        "--glitch-frac",
        "0.2",
        "--batches-per-epoch",
        "2",
        "--batch-size",
        "10",
        "--sample-rate",
        "256",
        "--max-epochs",
        "1",
        "resnet",
        "--layers",
        "2",
        "2",
    ]

    # since trainify wraps function w/ typeo
    # looks for args from command line
    # i.e. from sys.argv
    result = fn()
    assert len(result["train_loss"]) == 1
