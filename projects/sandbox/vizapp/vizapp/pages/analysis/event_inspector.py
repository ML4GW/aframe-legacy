import logging
import re
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Legend,
    LinearAxis,
    Range1d,
)
from bokeh.plotting import figure
from vizapp import palette

from aframe.analysis.ledger.injections import LigoResponseSet
from ml4gw.utils.slicing import unfold_windows

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


class EventInspectorPlot:
    def __init__(self, page):
        self.page = page
        # intialize some attrs that live in the app
        self.strain_dir = self.page.app.strain_dir
        self.fduration = self.page.app.fduration
        self.model = self.page.app.model
        self.background_length = self.page.app.background_length
        self.sample_rate = self.page.app.sample_rate
        self.kernel_length = self.page.app.kernel_length
        self.kernel_size = int(self.kernel_length * self.sample_rate)
        self.inference_stride = self.page.app.inference_stride
        self.ifos = self.page.app.ifos

        self.foreground = self.page.app.foreground

    def initialize_sources(self):
        self.strain_source = ColumnDataSource(dict(H1=[], L1=[], t=[]))
        self.response_source = ColumnDataSource(
            dict(nn=[], integrated=[], t=[])
        )

    def get_layout(self, height: int, width: int) -> None:
        self.timeseries_plot = figure(
            title="Click on an event to inspect",
            height=height,
            width=width,
            y_range=(-2, 2),
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

        return self.timeseries_plot

    def get_background_strain(
        self, time: float, pad: int, shifts: Tuple[float, float]
    ):
        # query the background strain including data from the shift
        fname, t0, duration = get_strain_fname(self.strain_dir, time)
        times = np.arange(t0, t0 + duration, 1 / self.sample_rate)

        start, stop = get_indices(
            times,
            time - self.kernel_length - pad,
            time + self.fduration + 1 + pad,
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

    def inject(self, strain, time, shifts):
        # find the closest injection that corresponds to event
        # time and shifts and read it in
        waveform = LigoResponseSet.read(
            self.page.app.response_path, time - 1000, time + 1000, shifts
        )
        if len(waveform) == 0:
            return

        waveform = torch.stack(
            [
                torch.tensor(getattr(waveform, ifo.lower())[0])
                for ifo in self.ifos
            ],
            axis=0,
        )
        waveform *= 2
        # reduce waveform length to 2 seconds
        waveform_length = waveform.shape[-1] / self.sample_rate
        idx = int(self.sample_rate * (waveform_length - 2) // 2)
        waveform = waveform[:, idx:-idx]

        # inject waveform into strain
        waveform_size = waveform.shape[-1]
        waveform_start = int(
            (self.kernel_size + (4 * self.sample_rate)) - (waveform_size / 2)
        )
        waveform_stop = waveform_start + waveform_size
        strain[:, waveform_start:waveform_stop] += waveform
        return strain

    def analyze_strain(self, strain):
        batch = unfold_windows(strain, self.kernel_size, self.inference_stride)
        whitened = self.page.app.preprocessor(batch)
        outputs = self.model(whitened)
        return outputs, whitened

    def update(
        self,
        event_time: float,
        event_type: str,
        shift: np.ndarray,
        title: str,
    ) -> None:

        event_time = 1262844473 + 6000
        foreground = event_type == "foreground"
        pad = 4
        strain, times = self.get_background_strain(event_time, pad, shift)
        if foreground:
            strain = self.inject(strain, event_time, shift)
            if strain is None:
                logging.warning(
                    "No waveform found for event time {}".format(event_time)
                )

        nn, whitened = self.analyze_strain(strain)

        # times refer to front of window
        start = int(
            (self.kernel_length - (self.fduration // 2)) * self.sample_rate
        )
        inference_times = times[start :: self.inference_stride]

        # normalize times with respect to event time
        times = times - event_time
        inference_times = inference_times - event_time
        nn = nn.detach().numpy().flatten()
        # self.strain_source.data = {"H1": h1, "L1": l1, "t": times}

        # for r in self.strain_renderers:
        # r.data_source.data = {"H1": h1, "L1": l1, "t": times}

        self.response_source.data = {
            "nn": nn,
            "t": inference_times,
        }
        for r in self.output_renderers:
            r.data_source.data = dict(nn=nn, t=inference_times)

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
