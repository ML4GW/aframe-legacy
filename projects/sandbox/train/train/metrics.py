import torch
from typing import List, Sequence
import numpy as np

class Metric(torch.nn.Module):
    """
    Abstract class for representing a metric which
    needs to be evaluated at particular threshold(s).
    Inherits from `torch.nn.Module` so that parameters
    for calculating the metric of interest can be
    saved as `buffer`s and moved to appropriate devices
    via `Metric.to`. Child classes should override `call`
    for actually computing the metric values of interest.
    """

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
        self.values = [v for v in values.cpu().numpy()]
        return values

    def __str__(self):
        tab = " " * 8
        string = ""
        for threshold, value in zip(self.thresholds, self.values):
            string += f"\n{tab}{self.param} = {threshold}: {value:0.7f}"
        return self.name + " @:" + string

    def __getitem__(self, threshold):
        try:
            idx = self.thresholds.index(threshold)
        except ValueError:
            raise KeyError(str(threshold))
        return self.values[idx]

    def __contains__(self, threshold):
        return threshold in self.thresholds


class MultiThresholdAUROC(Metric):
    name = "AUROC"
    param = "max_fpr"

    def call(self, signal_preds, background_preds):
        x = torch.cat([signal_preds, background_preds])
        y = torch.zeros_like(x)
        thresholds = torch.Tensor(self.thresholds).to(y.device)
        y[: len(signal_preds)] = 1

        # shuffle the samples so that constant
        # network outputs don't show up as perfect
        idx = torch.randperm(len(y))
        x = x[idx]
        y = y[idx]

        # now sort the labels by their corresponding prediction
        idx = torch.argsort(x, descending=True)
        y = y[idx]

        tpr = torch.cumsum(y, -1) / y.sum()
        fpr = torch.cumsum(1 - y, -1) / (1 - y).sum()
        dfpr = fpr.diff()
        dtpr = tpr.diff()

        mask = fpr[:-1, None] <= thresholds
        dfpr = dfpr[:, None] * mask
        integral = (tpr[:-1, None] + dtpr[:, None] * 0.5) * dfpr
        return integral.sum(0)


class BackgroundAUROC(MultiThresholdAUROC):
    def __init__(
        self, kernel_size: int, stride: int, thresholds: List[float]
    ) -> None:
        super().__init__(thresholds)
        self.kernel_size = kernel_size
        self.stride = stride

    def call(self, background, _, signal):
        return super().call(signal, background)


class BackgroundRecall(Metric):
    """
    Computes the recall of injected signals (fraction
    of total injected signals recovered) at the
    detection statistic threshold given by each of the
    top `k` background "events."

    Background predictions are max-pooled along the time
    dimension using the indicated `kernel_size` and
    `stride` to keep from counting multiple events from
    the same phenomenon.

    Args:
        kernel_size:
            Size of the window, in samples, over which
            to max pool background predictions.
        stride:
            Number of samples between consecutive
            background max pool windows.
        k:
            Max number of top background events against
            whose thresholds to evaluate signal recall.
    """

    name = "recall vs. background"
    param = "k"

    def __init__(self, kernel_size: int, stride: int, k: int = 5) -> None:
        super().__init__([i + 1 for i in range(k)])
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


class WeightedEfficiency(Metric):
    name = "Weighted efficiency"

    def __init__(self, weights: List[np.ndarray]) -> None:
        super().__init__(weights)

    def call(self, background_events, foreground_events):
        threshold = background_events.max()
        mask = foreground_events >= threshold
        effs = self.weights[mask].sum() / self.weights.sum()
        return effs


class GlitchRecall(Metric):
    """
    Computes the recall of injected signals (fraction
    of total injected signals recovered) at the detection
    statistic threshold given by each of the glitch
    specificity values (fraction of glitches rejected)
    specified.

    Args:
        specs:
            Glitch specificity values against which to
            compute detection statistic thresholds.
            Represents the fraction of glitches that would
            be rejected at a given threshold.
    """

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