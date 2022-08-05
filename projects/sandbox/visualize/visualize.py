import re
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import numpy as np
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Colorblind8 as palette
from bokeh.plotting import figure
from gwpy.timeseries import TimeSeries

from bbhnet.analysis.distributions.cluster import ClusterDistribution
from bbhnet.io.timeslide import Timeslide
from hermes.typeo import typeo


def get_training_curves(output_directory: Path):
    """
    Read the training logs from the indicated output directory
    and use them to parse the training and validation loss values
    at each epoch. Plot these curves to a Bokeh `Figure` and return it.
    """

    with open(output_directory / "train.log", "r") as f:
        train_log = f.read()

    epoch_re = re.compile("(?<==== Epoch )[0-9]{1,4}")
    train_loss_re = re.compile(r"(?<=Train Loss: )[0-9.e\-+]+")
    valid_loss_re = re.compile(r"(?<=Valid Loss: )[0-9.e\-+]+")

    source = ColumnDataSource(
        dict(
            epoch=list(map(int, epoch_re.findall(train_log))),
            train=list(map(float, train_loss_re.findall(train_log))),
            valid=list(map(float, valid_loss_re.findall(train_log))),
        )
    )

    p = figure(
        height=300,
        width=600,
        # sizing_mode="scale_width",
        title="Training curves",
        x_axis_label="Epoch",
        y_axis_label="ASDR",
        tools="reset,box_zoom",
    )

    r = p.line(
        "epoch",
        "train",
        line_width=2.3,
        line_color=palette[-1],
        line_alpha=0.8,
        legend_label="Train Loss",
        source=source,
    )
    p.line(
        "epoch",
        "valid",
        line_width=2.3,
        line_color=palette[-2],
        line_alpha=0.8,
        legend_label="Valid Loss",
        source=source,
    )

    p.add_tools(
        HoverTool(
            mode="vline",
            line_policy="nearest",
            point_policy="snap_to_data",
            renderers=[r],
            tooltips=[
                ("Epoch", "@epoch"),
                ("Train ASDR", "@train"),
                ("Valid ASDR", "@valid"),
            ],
        )
    )
    return p


def scan_events(
    n_events: float,
    ifos: Iterable[str],
    timeslide_dir: Path,
    distribution: ClusterDistribution,
    qrange: Tuple[float],
    frange: Tuple[float],
    tres: float,
    fres: float,
):

    # unpack the loudest and quietest n events
    events = distribution.events
    loudest_args = np.argsort(events)[-n_events:]
    loudest_times = distribution.event_times[loudest_args]
    loudest_shifts = distribution.shifts[loudest_args]

    # read in all segments
    ts = Timeslide(timeslide_dir / "dt-0.0-0.0", field="background")

    # for each event
    for time, shift in zip(loudest_times, loudest_shifts):

        for segment in ts.segments:

            # find segment that corresponds to shift
            shifted = ts.make_shift("dt-0.0-{float(shift)}")

            # if time is in this segment
            if time in shifted:

                # load in raw hoft data
                *raw, t = segment.load(*ifos)
                for i, ifo in ifos:

                    # create qscan
                    data = raw[i]
                    asd = data.asd()
                    timeseries = TimeSeries(data, times=t)
                    timeseries = timeseries.crop(time - 1, time + 1)
                    qscan = timeseries.q_transform(
                        frange=frange,
                        qrange=qrange,
                        tres=tres,
                        fres=fres,
                        whiten=asd,
                    )
                    qscan.plot()


@typeo
def main(
    results_dir: Path,
    timeslide_dir: Path,
    n_events: int,
    norm_seconds: Iterable[float],
    ifos: List[str],
    qrange: Tuple[float],
    frange: Tuple[float],
    fres: float,
    tres: float,
):

    # load in backgrounds
    # and injection results
    # for each normalizaiton

    for norm in norm_seconds:
        if norm == 0:
            norm = None
        injection_file = results_dir / f"injections-{norm}.h5"
        background_file = results_dir / f"background_{norm}.h5"

        with h5py.File(injection_file) as f:
            fars = f["fars"]
            latencies = f["latencies"]
            integrated = f["integrated"]

        background = ClusterDistribution.from_file(
            "integrated", background_file
        )

        # scan the loudest and quietest background events
        scan_events(
            n_events,
            ifos,
            timeslide_dir,
            background,
            qrange,
            frange,
            tres,
            fres,
        )
        return fars, latencies, integrated
