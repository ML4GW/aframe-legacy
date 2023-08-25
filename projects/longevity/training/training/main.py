from pathlib import Path
from typing import List, Optional

from train.train import main
from typeo import scriptify

from aframe.logging import configure_logging


@scriptify
def deploy_train(
    home: Path,
    ifos: List[str],
    # optimization args
    batch_size: int,
    snr_thresh: float,
    max_min_snr: float,
    max_snr: float,
    snr_alpha: float,
    snr_decay_steps: int,
    # data args
    sample_rate: float,
    kernel_length: float,
    psd_length: float,
    fduration: float,
    highpass: float,
    fftlength: Optional[float] = None,
    # augmentation args
    waveform_prob: float = 0.5,
    swap_frac: float = 0.0,
    mute_frac: float = 0.0,
    trigger_distance: float = 0,
    # validation args
    valid_frac: Optional[float] = None,
    valid_stride: Optional[float] = None,
    num_valid_views: int = 5,
    max_fpr: float = 1e-3,
    valid_livetime: float = (3600 * 12),
    early_stop: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
    # misc args
    device: str = "cpu",
    verbose: bool = False,
):

    configure_logging(home / "training.log", verbose=verbose)

    # for each interval, train a model
    intervals = [x for x in home.iterdir() if x.is_dir()]

    for interval in intervals:
        retrain_dir = interval / "retrained"
        retrain_dir.mkdir(exist_ok=True)

        datadir = retrain_dir / "data"
        background_dir = datadir / "train" / "background"
        waveform_dataset = datadir / "train" / "signals.h5"
        main(
            background_dir,
            waveform_dataset,
            retrain_dir / "training",
            retrain_dir / "log",
            ifos,
            batch_size,
            snr_thresh,
            max_min_snr,
            max_snr,
            snr_alpha,
            snr_decay_steps,
            sample_rate,
            kernel_length,
            psd_length,
            fduration,
            highpass,
            fftlength,
            waveform_prob,
            swap_frac,
            mute_frac,
            trigger_distance,
            valid_frac,
            valid_stride,
            num_valid_views,
            max_fpr,
            valid_livetime,
            early_stop,
            checkpoint_every,
            device,
            verbose,
        )
