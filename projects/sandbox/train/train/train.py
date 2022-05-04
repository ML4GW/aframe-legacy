from typing import Optional

from bbhnet.trainer.data import (
    GlitchSampler,
    RandomWaveformDataset,
    WaveformSampler,
)
from bbhnet.trainer.wrapper import trainify


# note that this function decorator acts both to
# wrap this function such that the outputs of it
# (i.e. the training and possible validation data)
# get passed as inputs to deepclean.trainer.trainer.train,
# as well as to expose these arguments _as well_ as those
# from deepclean.trainer.trainer.train to command line
# execution and parsing
@trainify
def main(
    output_directory: str,
    # data params
    glitch_dataset: str,
    signal_dataset: str,
    val_glitch_dataset: str,
    val_signal_dataset: str,
    hanford_background: str,
    livingston_background: str,
    val_livingston_background: str,
    val_hanford_background: str,
    waveform_frac: float,
    glitch_frac: float,
    sample_rate: float,
    min_snr: float,
    max_snr: float,
    highpass: float,
    device: str,
    kernel_length: float,
    batch_size: int,
    batches_per_epoch: int,
    valid_frac: Optional[float] = None,
    force_download: bool = False,
    verbose: bool = False,
    **kwargs
):

    # initiate training glitch sampler

    train_glitch_sampler = GlitchSampler(glitch_dataset, device=device)

    # initiate training waveform sampler
    train_waveform_sampler = WaveformSampler(
        signal_dataset, sample_rate, min_snr, max_snr, highpass, device=device
    )

    # deterministic validation glitch sampler
    # 'determinisitc' key word not yet implemented,
    # just an idea.
    val_glitch_sampler = GlitchSampler(
        val_glitch_dataset, device=device, deterministic=True, seed=100
    )

    # deterministic validation waveform sampler
    val_waveform_sampler = WaveformSampler(
        val_signal_dataset,
        sample_rate,
        highpass,
        device=device,
        deterministic=True,
        seed=100,
    )

    # create full training dataloader
    train_dataset = RandomWaveformDataset(
        hanford_background,
        livingston_background,
        kernel_length,
        sample_rate,
        batch_size,
        batches_per_epoch,
        train_waveform_sampler,
        waveform_frac,
        train_glitch_sampler,
        glitch_frac,
        device,
    )

    # create full validation dataloader
    valid_dataset = RandomWaveformDataset(
        val_hanford_background,
        val_livingston_background,
        kernel_length,
        sample_rate,
        batch_size,
        batches_per_epoch,
        val_waveform_sampler,
        waveform_frac,
        val_glitch_sampler,
        glitch_frac,
        device,
    )

    return train_dataset, valid_dataset
