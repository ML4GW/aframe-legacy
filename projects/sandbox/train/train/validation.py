import logging
import pickle
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np
import torch
from train.utils import split

if TYPE_CHECKING:
    from bbhnet.data.waveform_injection import BBHNetWaveformInjection


@dataclass
class Metric:
    @property
    def fields(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


@dataclass
class BackgroundRecall(Metric):
    kernel_size: int
    stride: int
    k: int = 5

    @property
    def fields(self):
        return [f"recall@k={i+1}" for i in range(self.k)]

    def __call__(self, background, signal):
        background = torch.nn.functional.max_pool1d(
            background, kernel_size=self.kernel_size, stride=self.stride
        )
        topk = torch.topk(background, self.k)
        recall = (signal.unsqueeze(1) >= topk).sum(0) / len(signal)
        return dict(zip(self.fields, recall.cpu().numpy()))


@dataclass
class GlitchRecall(Metric):
    specs: Sequence[float]

    def __post_init__(self):
        for spec in self.specs:
            assert 0 <= spec <= 1
        self.specs = torch.Tensor(self.specs)

    @property
    def fields(self):
        return [f"recall@spec={spec}" for spec in self.specs]

    def __call__(self, glitches, signal):
        qs = torch.quantile(glitches.unsqueeze(1), self.specs)
        recall = (signal.unsqueeze(1) >= qs).sum(0) / len(signal)
        return dict(zip(self.fields, recall.cpu().numpy()))


@dataclass
class Recorder:
    logdir: Path
    monitor: str
    kernel_length: int
    stride: int
    sample_rate: float
    topk: int = 5
    specs: Sequence[float] = [0.75, 0.9, 1]
    early_stop: Optional[int] = None
    checkpoint_every: Optional[int] = None

    def __post_init__(self):
        self.background = BackgroundRecall(
            int(self.kernel_length * self.sample_rate),
            int(self.stride * self.sample_rate),
            self.topk,
        )
        self.glitch = GlitchRecall(self.specs)

        fields = self.background.fields + self.glitch.fields
        if self.monitor not in fields:
            raise ValueError(
                f"Monitor field {self.monitor} not in metric fields {fields}"
            )
        self.history = {i: [] for i in fields + ["train_loss"]}

        if self.checkpoint_every is not None:
            (self.logdir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self.best = 0
        self._i = 0

    def record(
        self,
        model: torch.nn.Module,
        train_loss: float,
        background: torch.Tensor,
        glitches: torch.Tensor,
        signal: torch.Tensor,
    ):
        train_loss = train_loss
        metrics = {"train_loss": train_loss}
        bckgrd_recall = self.background(background, signal)
        glitch_recall = self.glitch(glitches, signal)

        metrics.update(bckgrd_recall)
        metrics.update(glitch_recall)

        msg = f"Train loss: {train_loss}"
        msg += "\nRecall vs. background @:"
        for field in self.background.fields:
            value = metrics[field]
            msg += "\n" + field.split("@")[1] + f": {value:0.3f}"
        msg += "\nRecall vs. glitches @:"
        for field in self.glitch.fields:
            value = metrics[field]
            msg += "\n" + field.split("@")[1] + f": {value:0.3f}"
        logging.info(msg)

        return self.checkpoint(model, metrics)

    def checkpoint(
        self, model: torch.nn.Module, metrics: Dict[str, float]
    ) -> bool:
        self._i += 1
        for key, val in metrics:
            self.history[key].append(val)

        if (
            self.checkpoint_every is not None
            and not self._i % self.checkpoint_every
        ):
            epoch = str(self._i).zfill(4)
            fname = self.logdir / "checkpoints" / f"epoch_{epoch}.pt"
            torch.save(model.state_dict(), fname)

        if metrics[self.monitor] > self.best:
            fname = self.logdir / "weights.pt"
            torch.save(model.state_dict(), fname)
            self._since_last = 0
        elif self.early_stop is not None:
            self._since_last += 1
            if self._since_last >= self.early_stop:
                return True
        return False


def make_background(
    background: np.ndarray, kernel_size: int, stride_size: int
) -> torch.Tensor:
    num_ifos, size = background.shape
    num_kernels = (size - kernel_size) // stride_size + 1
    num_kernels = int(num_kernels)

    background = background[:, : num_kernels * stride_size + kernel_size]
    background = torch.Tensor(background)[None, :, None]

    # fold out into windows up front
    background = torch.nn.functional.unfold(
        background, (1, num_kernels), dilation=(1, stride_size)
    )

    # some reshape magic having to do with how the
    # unfold op orders things. Don't worry about it
    background = background.reshape(num_ifos, num_kernels, kernel_size)
    background = background.transpose(1, 0)
    return background


def make_glitches(
    glitches: Sequence[np.ndarray],
    background: torch.Tensor,
    kernel_size: int,
    glitch_frac: float,
) -> torch.Tensor:
    if len(glitches) != background.size(1):
        raise ValueError(
            "Number of glitch tensors {} doesn't match number "
            "of interferometers {}".format(len(glitches), background.size(1))
        )

    h1_glitches, l1_glitches = map(torch.Tensor, glitches)
    num_h1, num_l1 = len(h1_glitches), len(l1_glitches)
    num_glitches = num_h1 + num_l1
    num_coinc = int(glitch_frac * num_glitches / (1 + glitch_frac))

    h1_coinc, h1_glitches = split(h1_glitches, num_coinc / num_h1, 0)
    l1_coinc, l1_glitches = split(l1_glitches, num_coinc / num_l1, 0)
    coinc = torch.stack([h1_coinc, l1_coinc], axis=1)
    num_h1, num_l1 = len(h1_glitches), len(l1_glitches)
    num_glitches = num_h1 + num_l1 + num_coinc

    # if we need to create duplicates of some of our
    # background to make this work, figure out how many
    background = repeat(background, num_glitches)

    # now insert the glitches
    start = h1_glitches.shape[-1] // 2 - kernel_size // 2
    slc = slice(start, start + kernel_size)

    background[:num_h1, 0] = h1_glitches[:, slc]
    background[num_h1:-num_coinc, 1] = l1_glitches[:, slc]
    background[-num_coinc:] = coinc[:, :, slc]

    return background


def repeat(X: torch.Tensor, max_num: int):
    repeats = ceil(len(X) / max_num)
    X = X.repeat(repeats, 1, 1)
    return X[:max_num]


class Validator:
    def __init__(
        self,
        recorder: Recorder,
        background: np.ndarray,
        glitches: Sequence[np.ndarray],
        injector: "BBHNetWaveformInjection",
        kernel_length: float,
        stride: float,
        sample_rate: float,
        batch_size: int,
        glitch_frac: float,
        device: str,
    ) -> None:
        self.device = device
        self.recorder = recorder

        kernel_size = int(kernel_length * sample_rate)
        stride_size = int(stride * sample_rate)

        # create a datset of pure background
        background = make_background(background, kernel_size, stride_size)
        self.background_loader = self.make_loader(background, batch_size)

        # now repliate that dataset but with glitches inserted
        # into either or both interferometer channels
        glitch_background = make_glitches(
            glitches, background, kernel_size, glitch_frac
        )
        self.glitch_loader = self.make_loader(glitch_background, batch_size)

        # 3. create a tensor of background with waveforms injected
        waveforms = injector.sample(-1)
        signal_background = repeat(background, len(waveforms))

        start = waveforms.shape[-1] // 2 - kernel_size // 2
        stop = start + kernel_size
        signal_background += waveforms[:, :, start:stop]
        self.signal_loader = self.make_loader(signal_background, batch_size)

    def make_loader(self, X: torch.Tensor, batch_size: int):
        dataset = torch.utils.data.Dataset(X)
        return torch.utils.data.DataLoader(
            dataset,
            pin_memory=True,
            batch_size=batch_size,
            pin_memory_device=self.device,
        )

    def get_predictions(self, loader, model):
        preds = []
        for (X,) in loader:
            X = X.to(self.device)
            y_hat = model(X)[:, 0]
            preds.append(y_hat)
        return torch.cat(preds)

    @torch.no_grad
    def __call__(self, model: torch.nn.Module, train_loss: float) -> bool:
        background_preds = self.get_predictions(self.background_loader, model)
        glitch_preds = self.get_predictions(self.glitch_loader, model)
        signal_preds = self.get_predictions(self.signal_loader, model)

        return self.recorder.record(
            model, train_loss, background_preds, glitch_preds, signal_preds
        )

    def save(self):
        with open(self.recorder.logdir / "history.pkl", "wb") as f:
            pickle.dump(self.recorder.history, f)
