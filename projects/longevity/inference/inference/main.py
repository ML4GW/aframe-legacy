import logging
from pathlib import Path
from typing import List

from infer.deploy import main as deploy_infer
from typeo import scriptify

from aframe.logging import configure_logging


@scriptify
def main(
    basedir: Path,
    original_model_repo_dir: str,
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
    configure_logging(basedir / "infer.log", verbose=verbose)
    # loop over intervals
    intervals = [x for x in basedir.iterdir() if x.is_dir()]

    for interval in intervals:
        datadir = interval / "data"
        injection_set_file = datadir / "test" / "waveforms.h5"

        # launch inference job for each interval
        # analyzing data with the retrained models
        retrained = interval / "retrained"
        infer_dir = retrained / "infer"
        infer_dir.mkdir(exist_ok=True, parents=True)
        complete = all(
            [
                f.exists()
                for f in [
                    infer_dir / "foreground.h5",
                    infer_dir / "background.h5",
                ]
            ]
        )
        if not complete:

            logging.info(
                "Deploying inference usiing "
                f"retrained model for {interval.name}"
            )
            deploy_infer(
                retrained / "model_repo",
                infer_dir,
                datadir / "test" / "background",
                retrained / "log",
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

        logging.info(f"inference complete for {interval.name}")
        # launch inference job for each interval
        # analyzing data with the original model
        original = interval / "original"
        infer_dir = original / "infer"
        infer_dir.mkdir(exist_ok=True, parents=True)

        complete = all(
            [
                f.exists()
                for f in [
                    infer_dir / "foreground.h5",
                    infer_dir / "background.h5",
                ]
            ]
        )
        if not complete:
            logging.info(
                "Deploying inference using original "
                f"model for {interval.name}"
            )
            deploy_infer(
                original_model_repo_dir,
                infer_dir,
                datadir / "test" / "background",
                original / "log",
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
