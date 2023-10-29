import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import toml
from typeo import scriptify

from aframe.logging import configure_logging

source_dir = Path(__file__).resolve().parent.parent
train_config_path = source_dir.parent / "pyproject.toml"


def read_config(path):
    with open(path, "r") as f:
        return toml.load(f)


@scriptify
def launch_train(interval: Path, gpu: int, seed: int):

    # construct directories / dataset paths for this interval
    retrain_dir = interval / "retrained"
    retrain_dir.mkdir(exist_ok=True, parents=True)

    datadir = interval / "data"
    background_dir = datadir / "train" / "background"
    waveform_dataset = datadir / "train" / "signals.h5"

    # read config that contains train arguments
    # and update paths for this directory
    config = read_config(train_config_path)
    train_config = config["tool"]["typeo"]["scripts"]["train"]

    train_config["background_dir"] = str(background_dir)
    train_config["waveform_dataset"] = str(waveform_dataset)
    train_config["logdir"] = str(retrain_dir / "log")
    train_config["outdir"] = str(retrain_dir / "training")
    train_config["seed"] = seed

    # write the config to the run's directory
    config_path = retrain_dir / "pyproject.toml"
    with open(config_path, "w") as f:
        toml.dump(config, f)

    # write the env file to the run's directory
    dotenv_path = retrain_dir / "train.env"
    with open(dotenv_path, "w") as f:
        f.write(f"export CUDA_VISIBLE_DEVICES={gpu}\n")

    cmd = [
        str(shutil.which("pinto")),
        "-p",
        str(source_dir),
        "run",
        "-e",
        str(dotenv_path),
        "train",
        "--typeo",
        f"{config_path}:train:resnet",
    ]

    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    # was getting weird pinto errors, so just set the path manually
    env[
        "PATH"
    ] = "/home/ethan.marx/.cache/pypoetry/virtualenvs/training-zaLaaJ_H-py3.9/bin"  # noqa
    subprocess.check_output(cmd, env=env)


@scriptify
def main(basedir: Path, gpus: List[int], seed: int):
    configure_logging(basedir / "train.log", verbose=True)
    # for each interval, train a model
    intervals = [
        x for x in basedir.iterdir() if x.is_dir() and x.name != "condor"
    ]
    futures = []
    with ThreadPoolExecutor(len(gpus)) as ex:
        for gpu, interval in zip(gpus, intervals):
            logging.info(f"Training for interval {interval} on GPU {gpu}")
            future = ex.submit(launch_train, interval, gpu, seed)
            futures.append(future)

    for f in as_completed(futures):
        gpu = f.result()
        logging.info(f"Finished training for interval {interval} on GPU {gpu}")
