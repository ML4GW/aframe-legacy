from typing import TYPE_CHECKING, Callable, Optional

from export.main import main as export
from typeo import scriptify

if TYPE_CHECKING:
    from pathlib import Path

    import hermes.quiver as qv


@scriptify
def main(
    basedir: Path,
    architecture: Callable,
    logdir: Path,
    num_ifos: int,
    kernel_length: float,
    inference_sampling_rate: float,
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
    intervals = [x for x in basedir.iterdir() if x.is_dir()]
    for interval in intervals:
        repository_directory = interval / "retrained" / "model_repo"
        export(
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
        )
