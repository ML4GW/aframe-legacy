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


class Metric(torch.nn.Module):
    def __init__(self, thresholds) -> None:
        super().__init__()
        self.thresholds = thresholds
        self.values = [0.0 for _ in thresholds]

    def update(self, metrics):
        try:
            metric = metrics[self.name]
        except KeyError:
            metric = {}
            metrics[self.name] = {}

        for threshold, value in zip(self.thresholds, self.values):
            try:
                metric[threshold].append(value)
            except KeyError:
                metric[threshold] = [value]

    def call(self, backgrounds, glitches, signals):
        raise NotImplementedError

    def forward(self, backgrounds, glitches, signals):
        values = self.call(backgrounds, glitches, signals)
        values = values.cpu().numpy()
        self.values = [v for v in values]

    def __str__(self):
        tab = " " * 8
        string = ""
        for threshold, value in zip(self.thresholds, self.values):
            string += f"\n{tab}{self.param} = {threshold}: {value:0.3f}"
        return self.name + " @:" + string

    def __getitem__(self, threshold):
        try:
            idx = self.thresholds.index(threshold)
        except ValueError:
            raise KeyError(str(threshold))
        return self.values[idx]

    def __contains__(self, threshold):
        return threshold in self.thresholds


class BackgroundRecall(Metric):
    name = "recall vs. background"
    param = "k"

    def __init__(self, kernel_size: int, stride: int, k: int = 5) -> None:
        super().__init__(list(range(k)))
        self.kernel_size = kernel_size
        self.stride = stride
        self.k = k

    def call(self, background, _, signal):
        background = background.unsqueeze(0)
        background = torch.nn.functional.max_pool1d(
            background, kernel_size=self.kernel_size, stride=self.stride
        )
        background = background[0]
        topk = torch.topk(background, self.k).values
        recall = (signal.unsqueeze(1) >= topk).sum(0) / len(signal)
        return recall


class GlitchRecall(Metric):
    name = "recall vs. glitches"
    param = "specificity"

    def __init__(self, specs: Sequence[float]) -> None:
        for i in specs:
            assert 0 <= i <= 1
        super().__init__(specs)
        self.register_buffer("specs", torch.Tensor(specs))

    def call(self, _, glitches, signal):
        qs = torch.quantile(glitches.unsqueeze(1), self.specs)
        recall = (signal.unsqueeze(1) >= qs).sum(0) / len(signal)
        return recall


@dataclass
class Recorder:
    logdir: Path
    monitor: Metric
    threshold: float
    additional: Optional[Sequence[Metric]] = None
    early_stop: Optional[int] = None
    checkpoint_every: Optional[int] = None

    def __post_init__(self):
        if self.threshold not in self.monitor:
            raise ValueError(
                "Metric {} has no threshold {}".format(
                    self.monitor.name, self.threshold
                )
            )
        self.history = {"train_loss": []}

        if self.checkpoint_every is not None:
            (self.logdir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self.best = -1  # best monitored metric value so far
        self._i = 0  # epoch counter
        self._since_last = 0  # epochs since last best monitored metric

    def record(
        self,
        model: torch.nn.Module,
        train_loss: float,
        background: torch.Tensor,
        glitches: torch.Tensor,
        signal: torch.Tensor,
    ):
        self.history["train_loss"].append(train_loss)
        self.monitor(background, glitches, signal)
        self.monitor.update(self.history)

        msg = f"Summary:\nTrain loss: {train_loss:0.3e}"
        msg += f"\nValidation {self.monitor}"
        if self.additional is not None:
            for metric in self.additional:
                metric(background, glitches, signal)
                metric.update(self.history)
                msg += f"\nValidation {metric}"
        logging.info(msg)

        return self.checkpoint(model, self.history)

    def checkpoint(
        self, model: torch.nn.Module, metrics: Dict[str, float]
    ) -> bool:
        self._i += 1
        with open(self.logdir / "history.pkl", "wb") as f:
            pickle.dump(self.history, f)

        if (
            self.checkpoint_every is not None
            and not self._i % self.checkpoint_every
        ):
            epoch = str(self._i).zfill(4)
            fname = self.logdir / "checkpoints" / f"epoch_{epoch}.pt"
            torch.save(model.state_dict(), fname)

        if self.monitor[self.threshold] > self.best:
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
    repeats = ceil(max_num / len(X))
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
        recorder.monitor.to(device)
        if recorder.additional is not None:
            for metric in recorder.additional:
                metric.to(device)
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
        waveforms, _ = injector.sample(-1)
        signal_background = repeat(background, len(waveforms))

        start = waveforms.shape[-1] // 2 - kernel_size // 2
        stop = start + kernel_size
        signal_background += waveforms[:, :, start:stop]
        self.signal_loader = self.make_loader(signal_background, batch_size)

    def make_loader(self, X: torch.Tensor, batch_size: int):
        dataset = torch.utils.data.TensorDataset(X)
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

    @torch.no_grad()
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
