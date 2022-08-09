import re
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Colorblind8 as palette
from bokeh.plotting import figure
from gwpy.plot import Plot
from gwpy.timeseries import TimeSeries

from bbhnet.analysis.distributions.cluster import ClusterDistribution
from bbhnet.io.timeslides import TimeSlide
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
    results_dir: Path,
    distribution: ClusterDistribution,
    kernel_length: float,
    qrange: Tuple[float],
    frange: Tuple[float],
    tres: float,
    fres: float,
    durs: List[float],
):

    """Generates omega scan of various durations
    for the most significant and quietest background
    events
    """

    # unpack the loudest n events
    events = distribution.events
    sorted_args = np.argsort(events)
    args = sorted_args[-n_events:]

    events = distribution.events[args]
    times = distribution.event_times[args]
    shifts = distribution.shifts[args]

    # read in all segments
    ts = TimeSlide(timeslide_dir / "dt-0.0-0.0", field="background")

    # for each event
    for time, shift, event in zip(times, shifts, events):

        for segment in ts.segments:

            # find segment that corresponds to shift
            shifted = segment.make_shift(f"dt-0.0-{float(shift)}")

            # if time is in this segment
            if time in shifted:

                # center time on middle of kernel
                time = time - kernel_length / 2

                # load in raw hoft data
                *raw, t = shifted.load(*ifos)
                qgrams = []
                for i, ifo in enumerate(ifos):

                    # create qscan
                    longest = max(durs)
                    data = raw[i]
                    # TODO: generalize this to multi ifo.
                    # should store shifts for all ifos
                    if i != 0:
                        times = times - shift
                        shifted_time = time - shift
                    else:
                        shifted_time = time
                    timeseries = TimeSeries(data, times=t)
                    asd = timeseries.asd()
                    timeseries = timeseries.crop(
                        shifted_time - longest - 1, shifted_time + longest + 1
                    )
                    # why was this failing again?
                    try:
                        qgrams.extend(
                            [
                                timeseries.q_transform(
                                    frange=frange,
                                    logf=True,
                                    outseg=(
                                        shifted_time - dur,
                                        shifted_time + dur,
                                    ),
                                    whiten=asd,
                                )
                                for dur in durs
                            ]
                        )
                    except ValueError:
                        continue

                fig = Plot(
                    *qgrams,
                    figsize=(8 * len(durs), 8),
                    geometry=(2, len(durs)),
                    yscale="log",
                    method="pcolormesh",
                )

                for i, ax in enumerate(fig.axes):
                    if i <= len(durs) - 1:
                        ax.set_epoch(time)
                    else:
                        ax.set_epoch(time - shift)
                    fig.colorbar(
                        ax=ax, label="Normalized energy", clim=(0, 30)
                    )

                fig.suptitle(
                    "Omegascans of background event"
                    f"w/ integrated output {event}",
                    fontweight="bold",
                )

                fig.savefig(
                    results_dir / f"background_scan_{event:.2f}.png",
                    format="png",
                )


def roc_curve(
    background: ClusterDistribution,
    inj_fars: np.ndarray,
    inj_snrs: np.ndarray,
    far_thresholds: np.ndarray,
    norm: float,
    outdir: Path,
):

    fig = plt.figure()

    # create bins
    n_bins = 10
    bins = np.logspace(0, np.log10(max(inj_snrs)), n_bins)
    mid = (bins[:-1] + bins[1:]) / 2

    for thresh in far_thresholds:

        effs = []
        for i in range(len(bins) - 1):

            # calculate efficiency for this
            # bin at far thresh
            mask = inj_snrs < bins[i + 1]
            mask &= inj_snrs > bins[i]
            fars = inj_fars[mask]
            eff = len(fars[fars < thresh]) / len(fars)
            effs.append(eff)

        plt.plot(mid, effs, label=f"FAR {thresh} yr^-1")
        plt.xlabel("Network SNR")
        plt.ylabel("Detection efficiency")
        plt.title(
            f"Detection efficiency vs SNR at fixed FAR "
            f"({norm} s normalization)"
        )

    plt.legend()
    fig.savefig(outdir / "roc.png")


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
    durs: List[float],
    kernel_length: float,
):

    # load in backgrounds
    # and injection results
    # for each normalizaiton

    for norm in norm_seconds:
        # 0 norm seconds corresponds to no normalization
        if norm == 0:
            norm = None

        norm_results_dir = results_dir / f"norm-{norm}"
        norm_results_dir.mkdir(exist_ok=True, parents=True)

        injection_file = results_dir / f"injections-{norm}.h5"
        background_file = results_dir / f"background_{norm}.h5"

        with h5py.File(injection_file) as f:
            inj_fars = f["fars"][()][:, -1]
            inj_snrs = np.sqrt(f["H1_snr"][()] ** 2 + f["L1_snr"][()] ** 2)

        background = ClusterDistribution.from_file(
            "integrated", background_file
        )

        # create roc curve at various fixed FAR rates
        far_thresholds = [1, 10, 1000, 10000, 100000]
        roc_curve(
            background,
            inj_fars,
            inj_snrs,
            far_thresholds,
            norm,
            norm_results_dir,
        )

        # scan the loudest and quietest background events
        scan_events(
            n_events,
            ifos,
            timeslide_dir,
            norm_results_dir,
            background,
            kernel_length,
            qrange,
            frange,
            tres,
            fres,
            durs,
        )

        # histogram background and injection
        # distributions

    return results_dir
