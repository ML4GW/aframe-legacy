import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from export.main import main as export

import hermes.quiver as qv
from aframe.architectures import architecturize
from aframe.logging import configure_logging


@architecturize
def main(
    architecture: Callable,
    basedir: Path,
    num_ifos: int,
    kernel_length: float,
    inference_sampling_rate: float,
    sample_rate: float,
    batch_size: int,
    fduration: float,
    psd_length: float,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    streams_per_gpu: int = 1,
    aframe_instances: Optional[int] = None,
    platform: qv.Platform = qv.Platform.ONNX,
    clean: bool = False,
    verbose: bool = False,
):
    configure_logging(basedir / "export.log")
    intervals = [x for x in basedir.iterdir() if x.is_dir()]
    with ThreadPoolExecutor(8) as executor:
        for interval in intervals:
            logging.info(f"Exporting for interval {interval}")
            repository_directory = interval / "retrained" / "model_repo"
            logdir = interval / "retrained" / "log"
            weights = interval / "retrained" / "training" / "weights.pt"
            args = [
                architecture,
                repository_directory,
                logdir,
                num_ifos,
                kernel_length,
                inference_sampling_rate,
                sample_rate,
                batch_size,
                fduration,
                psd_length,
                fftlength,
                highpass,
                weights,
                streams_per_gpu,
                aframe_instances,
                platform,
                clean,
                verbose,
            ]
            executor.submit(export, *args)
