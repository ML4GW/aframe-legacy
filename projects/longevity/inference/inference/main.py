import logging
from pathlib import Path
from typing import List

from infer.deploy import main as deploy_infer
from typeo import scriptify

from aframe.logging import configure_logging


@scriptify
def main(
    datadir: Path,
    model_repo_dir: str,
    image: str,
    model_name: str,
    accounting_group: str,
    accounting_group_user: str,
    Tb: float,
    shifts: List[float],
    sample_rate: float,
    inference_sampling_rate: float,
    ifos: List[str],
    batch_size: int,
    integration_window_length: float,
    cluster_window_length: float,
    psd_length: float,
    fduration: float,
    throughput: float,
    chunk_size: float,
    sequence_id: int,
    model_version: int = -1,
    verbose: bool = False,
):
    configure_logging(datadir / "infer.log", verbose=verbose)
    # loop over intervals
    intervals = [x for x in datadir.iterdir() if x.is_dir()]
    for subdir in intervals:
        log_dir = subdir / "logs"
        output_dir = subdir / "infer"
        data_dir = subdir / "test" / "background"
        injection_set_file = subdir / "test" / "waveforms.h5"

        if injection_set_file.exists():
            logging.info(f"Deploying inference for {subdir.name}")
            deploy_infer(
                model_repo_dir,
                output_dir,
                data_dir,
                log_dir,
                injection_set_file,
                image,
                model_name,
                accounting_group,
                accounting_group_user,
                Tb,
                shifts,
                sample_rate,
                inference_sampling_rate,
                ifos,
                batch_size,
                integration_window_length,
                cluster_window_length,
                psd_length,
                fduration,
                throughput,
                chunk_size,
                sequence_id,
                model_version,
                verbose,
            )
