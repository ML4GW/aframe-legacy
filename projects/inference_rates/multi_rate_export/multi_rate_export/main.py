import logging
from pathlib import Path
from typing import Callable, List, Optional

from export.main import main as export

import hermes.quiver as qv
from aframe.architectures import architecturize
from aframe.logging import configure_logging


@architecturize
def main(
    architecture: Callable,
    repository_directory: Path,
    logdir: Path,
    num_ifos: int,
    kernel_length: float,
    inference_sampling_rates: List[float],
    sample_rate: float,
    batch_size: int,
    fduration: float,
    psd_length: float,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    weights: Optional[Path] = None,
    streams_per_gpu: int = 1,
    aframe_instances: Optional[int] = None,
    platform: qv.Platform = qv.Platform.ONNX,
    clean: bool = False,
    verbose: bool = False,
):
    configure_logging(logdir / "inference_rates_export.log", verbose=verbose)

    for rate in inference_sampling_rates:
        current_repo_dir = repository_directory / f"{rate}Hz"

        logging.info(f"Starting export with inference rate {rate} Hz")

        export(
            architecture=architecture,
            repository_directory=current_repo_dir,
            logdir=logdir,
            num_ifos=num_ifos,
            kernel_length=kernel_length,
            inference_sampling_rate=rate,
            sample_rate=sample_rate,
            batch_size=batch_size,
            fduration=fduration,
            psd_length=psd_length,
            fftlength=fftlength,
            highpass=highpass,
            weights=weights,
            streams_per_gpu=streams_per_gpu,
            aframe_instances=aframe_instances,
            platform=platform,
            clean=clean,
            verbose=verbose,
        )

        logging.info(f"Finished export with inference rate {rate} Hz")
