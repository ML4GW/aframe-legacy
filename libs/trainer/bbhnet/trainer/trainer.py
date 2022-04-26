import logging
import time
from typing import Optional

import torch

from bbhnet.data import RandomWaveformDataset


def train_for_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_data: RandomWaveformDataset,
    valid_data: RandomWaveformDataset = None,
    profiler: Optional[torch.profiler.profile] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    """Run a single epoch of training"""

    train_loss = 0
    samples_seen = 0
    start_time = time.time()
    model.train()

    for samples, targets in train_data:
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
    if valid_data is not None:
        valid_loss = 0
        samples_seen = 0

        model.eval()

        # reason mixed precision is not used here?
        with torch.no_grad():
            for samples, targets in valid_data:

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
