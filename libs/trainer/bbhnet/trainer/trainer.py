import logging
import os
import time
from typing import Callable, Optional

import numpy as np
import torch

from bbhnet.data import GlitchSampler, RandomWaveformDataset, WaveformSampler


def train_for_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_dataset: RandomWaveformDataset,
    valid_dataset: Optional[RandomWaveformDataset] = None,
    profiler: Optional[torch.profiler.profile] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    """Run a single epoch of training"""

    train_loss = 0
    samples_seen = 0
    start_time = time.time()
    model.train()

    for samples, targets in train_dataset:
        optimizer.zero_grad(set_to_none=True)  # reset gradient

        # do forward step in mixed precision
        with torch.autocast("cuda"):
            predictions = model(samples)
            loss = criterion(predictions, targets)

        train_loss += loss.item()
        samples_seen += len(samples)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if profiler is not None:
            profiler.step()

    if profiler is not None:
        profiler.stop()

    end_time = time.time()
    duration = end_time - start_time
    throughput = samples_seen / duration
    train_loss /= samples_seen

    logging.info(
        "Duration {:0.2f}s, Throughput {:0.1f} samples/s".format(
            duration, throughput
        )
    )
    msg = f"Train Loss: {train_loss:.4e}"

    # Evaluate performance on validation set if given
    if valid_dataset is not None:
        valid_loss = 0
        samples_seen = 0

        model.eval()

        # reason mixed precision is not used here?
        # since no gradient calculation that requires
        # higher precision?
        with torch.no_grad():
            for samples, targets in valid_dataset:

                predictions = model(samples)
                loss = criterion(predictions, targets)

                valid_loss += loss.item()
                samples_seen += len(samples)

        valid_loss /= samples_seen
        msg += f", Valid Loss: {valid_loss:.4e}"
    else:
        valid_loss = None

    logging.info(msg)
    return train_loss, valid_loss, duration, throughput


def train(
    architecture: Callable,
    output_directory: str,
    # data params
    train_files: dict,
    val_files: dict,
    waveform_frac: float,
    glitch_frac: float,
    sample_rate: float,
    min_snr: float,
    max_snr: float,
    highpass: float,
    kernel_length: float,
    batch_size: int,
    batches_per_epoch: int,
    valid_frac: Optional[float] = None,
    # optimization params
    max_epochs: int = 40,
    init_weights: Optional[str] = None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: Optional[int] = None,
    factor: float = 0.1,
    early_stop: int = 20,
    # misc params
    device: Optional[str] = None,
    profile: bool = False,
) -> float:
    # TODO: add documentation

    os.makedirs(output_directory, exist_ok=True)

    # initiate training glitch sampler
    train_glitch_sampler = GlitchSampler(
        train_files["glitch dataset"], device=device
    )

    # initiate training waveform sampler
    train_waveform_sampler = WaveformSampler(
        train_files["signal dataset"],
        sample_rate,
        min_snr,
        max_snr,
        highpass,
        device=device,
    )

    # deterministic validation glitch sampler
    # 'determinisitc' key word not yet implemented,
    # just an idea.
    val_glitch_sampler = GlitchSampler(
        val_files["glitch dataset"],
        device=device,
        deterministic=True,
        seed=100,
    )

    # deterministic validation waveform sampler
    val_waveform_sampler = WaveformSampler(
        val_files["signal dataset"],
        sample_rate,
        highpass,
        device=device,
        deterministic=True,
        seed=100,
    )

    # create full training dataloader
    train_dataset = RandomWaveformDataset(
        train_files["hanford background"],
        train_files["livingston background"],
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
        val_files["hanford background"],
        val_files["livingston background"],
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

    # Creating model, loss function, optimizer and lr scheduler
    logging.info("Building and initializing model")

    # TODO: generalize to arbitrary architectures /
    # architecture parameters
    model = architecture()
    model.to(device)

    if init_weights is not None:
        # allow us to easily point to the best weights
        # from another run of this same function
        if os.path.isdir(init_weights):
            init_weights = os.path.join(init_weights, "weights.pt")

        logging.debug(
            f"Initializing model weights from checkpoint '{init_weights}'"
        )
        model.load_state_dict(torch.load(init_weights))
    logging.info(model)

    logging.info("Initializing loss and optimizer")

    # TODO: Allow different loss functions or
    # optimizers to be passed?

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    if patience is not None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=patience,
            factor=factor,
            threshold=0.0001,
            min_lr=lr * factor**2,
            verbose=True,
        )

    # start training
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()
    best_valid_loss = np.inf
    since_last_improvement = 0
    history = {"train_loss": [], "valid_loss": []}

    logging.info("Beginning training loop")
    for epoch in range(max_epochs):
        if epoch == 0 and profile:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=10),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(output_directory, "profile")
                ),
            )
            profiler.start()
        else:
            profiler = None

        logging.info(f"=== Epoch {epoch + 1}/{max_epochs} ===")
        train_loss, valid_loss, duration, throughput = train_for_one_epoch(
            model,
            optimizer,
            criterion,
            train_dataset,
            valid_dataset,
            profiler,
            scaler,
        )
        history["train_loss"].append(train_loss)

        # do some house cleaning with our
        # validation loss if we have one
        if valid_loss is not None:
            history["valid_loss"].append(valid_loss)

            # update our learning rate scheduler if we
            # indicated a schedule with `patience`
            if patience is not None:
                lr_scheduler.step(valid_loss)

            # save this version of the model weights if
            # we achieved a new best loss, otherwise check
            # to see if we need to early stop based on
            # plateauing validation loss
            if valid_loss < best_valid_loss:
                logging.debug(
                    "Achieved new lowest validation loss, "
                    "saving model weights"
                )
                best_valid_loss = valid_loss

                weights_path = os.path.join(output_directory, "weights.pt")
                torch.save(model.state_dict(), weights_path)
                since_last_improvement = 0
            else:
                since_last_improvement += 1
                if since_last_improvement >= early_stop:
                    logging.info(
                        "No improvement in validation loss in {} "
                        "epochs, halting training early".format(early_stop)
                    )
                    break

    return history
