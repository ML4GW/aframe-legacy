import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from bokeh.layouts import row
from bokeh.models import (
    ColumnDataSource,
    Div,
    HoverTool,
    Legend,
    LinearAxis,
    Range1d,
)
from bokeh.plotting import figure
from gwpy.timeseries import TimeSeries
from vizapp import palette

from aframe.analysis.ledger.injections import LigoResponseSet

t0_pattern = re.compile(r"[0-9]{10}(\.[0-9])?(?=-)")
dur_pattern = re.compile("[0-9]{2,8}(?=.hdf5)")


def get_indices(t: np.ndarray, lower: float, upper: float):
    mask = (lower <= t) & (t < upper)
    idx = np.where(mask)[0]
    return idx[0], idx[-1]


def get_strain_fname(data_dir: Path, event_time: float):
    for fname in data_dir.iterdir():
        try:
            t0 = float(t0_pattern.search(fname.name).group(0))
            dur = float(dur_pattern.search(fname.name).group(0))
        except AttributeError:
            continue

        if t0 <= event_time < (t0 + dur):
            break
    else:
        raise ValueError(
            "No file containing event time {} in "
            "data directory {}".format(event_time, data_dir)
        )
    return fname, t0, dur


@dataclass
class EventAnalyzer:
    model: torch.nn.Module
    preprocessor: torch.nn.Module
    strain_dir: Path
    qscan_dir: Path
    response_path: Path
    fduration: float
    kernel_length: float
    inference_sampling_rate: float
    ifos: List[str]
    background_length: float
    sample_rate: float
    padding: float
    integration_length: float
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        self.kernel_size = int(self.kernel_length * self.sample_rate)
        # amount of data before time of event to load to ensure
        # we can analyze pad seconds before event time
        self.length_previous_data = (
            self.padding
            + self.background_length
            + self.kernel_length
            + self.fduration / 2
            + self.integration_length
        )
        self.size = int(self.length_previous_data * self.sample_rate)
        self.inference_stride = int(
            self.sample_rate / self.inference_sampling_rate
        )
        self.integration_size = int(
            self.integration_length * self.inference_sampling_rate
        )
        self.window = (
            torch.ones((1, 1, self.integration_size)) / self.integration_size
        )
        self.window = self.window.to(self.device)

    def find_strain(self, time: float, shifts: np.ndarray):
        # query the background strain
        fname, t0, duration = get_strain_fname(self.strain_dir, time)
        times = np.arange(t0, t0 + duration, 1 / self.sample_rate)
        start, stop = get_indices(
            times,
            time - self.length_previous_data,
            time + self.padding + (self.fduration / 2),
        )
        times = times[start:stop]
        strain = []
        with h5py.File(fname, "r") as f:
            for ifo, shift in zip(self.ifos, shifts):
                shift_size = int(shift * self.sample_rate)
                start_shifted, stop_shifted = (
                    start + shift_size,
                    stop + shift_size,
                )
                data = torch.tensor(f[ifo][start_shifted:stop_shifted])
                strain.append(data)

        return torch.stack(strain, axis=0), times

    def find_waveform(self, time: float, shifts: np.ndarray):
        # find the closest injection that corresponds to event
        # time and shifts from waveform dataset
        waveform = LigoResponseSet.read(
            self.response_path, time - 0.1, time + 0.1, shifts
        )

        waveform = torch.stack(
            [
                torch.tensor(getattr(waveform, ifo.lower())[0])
                for ifo in self.ifos
            ],
            axis=0,
        )

        # reduce waveform length to 2 seconds
        waveform_length = waveform.shape[-1] / self.sample_rate
        idx = int(self.sample_rate * (waveform_length - 2) // 2)
        waveform = waveform[:, idx:-idx]
        return waveform

    def inject(self, strain, waveform):
        # inject waveform into strain
        waveform_size = waveform.shape[-1]
        waveform_start = int(self.size - (waveform_size / 2))
        waveform_stop = waveform_start + waveform_size
        strain[:, waveform_start:waveform_stop] += waveform
        return strain

    def qscan(self, strain):
        ts = TimeSeries(strain, sample_rate=self.sample_rate)
        qscan = ts.q_transform(logf=True, frange=(32, 512), whiten=False)
        return qscan

    def integrate(self, nn):
        nn = nn.squeeze(-1)[None]
        preds = torch.nn.functional.conv1d(nn, self.window)
        return preds

    def __call__(self, time: float, shifts: np.ndarray, foreground: bool):
        strain, times = self.find_strain(time, shifts)
        if foreground:
            waveform = self.find_waveform(time, shifts)
            strain = self.inject(strain, waveform)
        strain = strain[None]

        splits = [strain.shape[-1] - self.preprocessor.size]
        print(np.split(times, splits))
        _, times = np.split(times, splits)
        fduration_size = int(self.fduration * self.sample_rate / 2)
        times = times[fduration_size:-fduration_size]
        print(times.shape)
        windows, whitened = self.preprocessor(strain)
        whitened = whitened.detach().numpy()[0]

        # TODO: find window where trigger is center
        # center = whitened[0][0].detach().numpy()
        # qscan = self.qscan(center)
        # qscan_path = self.qscan_dir / "qscan.png"
        # plot = qscan.plot()
        # plot.save(qscan_path)

        outputs = self.model(windows)
        integrated = self.integrate(outputs)

        outputs = outputs.detach().numpy().flatten()
        integrated = integrated.detach().numpy().flatten()

        outputs = outputs[self.integration_size :]
        integrated = integrated[:-1]

        inference_times = (
            times[self.kernel_size :: self.inference_stride][
                : -self.integration_size + 1
            ]
            + self.kernel_length
        )

        return outputs, integrated, whitened, times, inference_times


class EventInspectorPlot:
    def __init__(self, page, analyzer: EventAnalyzer):
        self.page = page
        self.analyzer = analyzer

    def initialize_sources(self):
        self.strain_source = ColumnDataSource(dict(H1=[], L1=[], t=[]))
        self.response_source = ColumnDataSource(
            dict(nn=[], integrated=[], t=[])
        )
        self.spectrogram = Div(text="", width=400, height=400)

    def get_layout(self, height: int, width: int) -> None:
        self.timeseries_plot = figure(
            title="Click on an event to inspect",
            height=height,
            width=width,
            y_range=(-5, 5),
            x_range=(-3, 3),
            x_axis_label="Time [s]",
            y_axis_label="Strain [unitless]",
            tools="",
        )
        self.timeseries_plot.toolbar.autohide = True

        items, self.strain_renderers = [], []
        for i, ifo in enumerate(["H1", "L1"]):
            r = self.timeseries_plot.line(
                x="t",
                y=ifo,
                line_color=palette[i],
                line_alpha=0.6,
                # legend_label=ifo,
                source=self.strain_source,
            )
            self.strain_renderers.append(r)
            items.append((ifo, [r]))

        self.timeseries_plot.extra_y_ranges = {"nn": Range1d(-1, 10)}
        self.timeseries_plot.add_layout(
            LinearAxis(axis_label="NN output", y_range_name="nn"), "right"
        )

        self.output_renderers = []
        for i, field in enumerate(["nn", "integrated"]):
            label = "NN output"
            if field == "integrated":
                label = "Integrated " + label

            r = self.timeseries_plot.line(
                "t",
                field,
                line_color=palette[2 + i],
                line_width=2,
                line_alpha=0.8,
                # legend_label=label,
                source=self.response_source,
                y_range_name="nn",
            )
            self.output_renderers.append(r)
            items.append((label, [r]))

        legend = Legend(items=items, orientation="horizontal")
        self.timeseries_plot.add_layout(legend, "below")

        hover = HoverTool(
            renderers=[r],
            tooltips=[
                ("NN response", "@nn"),
                ("Integrated NN response", "@integrated"),
            ],
        )
        self.timeseries_plot.add_tools(hover)
        self.timeseries_plot.legend.click_policy = "mute"

        return row(self.timeseries_plot, self.spectrogram)

    def update(
        self,
        event_time: float,
        event_type: str,
        shift: np.ndarray,
        title: str,
    ) -> None:

        foreground = event_type == "foreground"
        nn, integrated, whitened, times, inference_times = self.analyzer(
            event_time, shift, foreground
        )

        # self.spectrogram.text = f"<img src=myapp/{qscan_path}>"

        h1, l1 = whitened
        # normalize times with respect to event time
        times = times - event_time
        inference_times = inference_times - event_time

        self.strain_source.data = {"H1": h1, "L1": l1, "t": times}
        for r in self.strain_renderers:
            r.data_source.data = {"H1": h1, "L1": l1, "t": times}

        self.response_source.data = {
            "nn": nn,
            "integrated": integrated,
            "t": inference_times,
        }
        for r in self.output_renderers:
            r.data_source.data = dict(
                nn=nn, integrated=integrated, t=inference_times
            )

        nn_min = nn.min()
        nn_max = nn.max()
        nn_min = 0.95 * nn_min if nn_min > 0 else 1.05 * nn_min

        nn_max = 1.05 * nn_max if nn_max > 0 else 0.95 * nn_max

        self.timeseries_plot.extra_y_ranges["nn"].start = nn_min
        self.timeseries_plot.extra_y_ranges["nn"].end = nn_max

        self.timeseries_plot.xaxis.axis_label = (
            f"Time from {event_time:0.3f} [s]"
        )

        self.timeseries_plot.title.text = title

    def reset(self):
        self.configure_sources()
        for r in self.strain_renderers:
            r.data_source.data = dict(H1=[], L1=[], t=[])

        for r in self.output_renderers:
            r.data_source.data = dict(nn=[], integrated=[], t=[])

        self.timeseries_plot.title.text = "Click on an event to inspect"
        self.timeseries_plot.xaxis.axis_label = "Time [s]"
