import datetime
from pathlib import Path
from typing import List

import h5py
import numpy as np
from bokeh.io import export_svg
from bokeh.models import Label, Legend, LegendItem, Span
from bokeh.palettes import RdYlBu4 as palette
from bokeh.plotting import figure, save


def get_figure(**kwargs):
    default_kwargs = dict(height=300, width=700, tools="")
    kwargs = default_kwargs | kwargs
    p = figure(**kwargs)

    if not kwargs.get("tools"):
        p.toolbar_location = None

    title = kwargs.get("title")
    if title and title.startswith("$$"):
        p.title.text_font_style = "normal"
    return p


def load_sv(path: Path, combo: str):
    with h5py.File(path, "r") as f:
        fars = f["fars"][:]
        combo = f[combo]
        sv = combo["sv"][:]
    return sv, fars


def calc_sv_over_time(
    intervals: List[Path], name: str, mass_combo: str, fars: np.ndarray
):
    data = []
    for interval in intervals:
        path = interval / name / "results" / "sensitive-volume.h5"
        sv, far = load_sv(path, mass_combo)
        interped = np.interp(fars, far, sv)
        data.append(interped)
    return data


def main(
    basedir: Path,
    original_path: Path,
    mass_combo: str,
    weeks: List[int],
):
    fars = np.array([0.5, 1, 2, 5])
    intervals = [
        x for x in basedir.iterdir() if x.is_dir() and x.name != "condor"
    ]
    intervals.sort(
        key=lambda x: datetime.datetime.strptime(
            x.name.split("_")[0], "%m-%d-%Y"
        )
    )

    # load in results for intervals
    original = calc_sv_over_time(intervals, "original", mass_combo, fars)

    # load in results for original model
    weeks.insert(0, 0)
    sv, far = load_sv(original_path, mass_combo)
    interped = np.interp(fars, far, sv)
    original.insert(0, interped)
    original = np.array(original)

    p = get_figure(
        title=f"Sensitive Distance Over Time for {mass_combo} Lognormal",
        x_axis_label="Weeks After Original Test Period",
        y_axis_label="Sensitive Distance (Mpc)",
        tools="save",
        output_backend="svg",
    )

    o3b_start = 1256655618
    o3a_end = 1253977218
    original_start = 1244035783
    ONE_WEEK = 60 * 60 * 24 * 7
    break_weeks = []
    for time in [o3a_end, o3b_start]:
        diff = time - original_start
        break_weeks.append(diff / ONE_WEEK)
    mid = np.mean(break_weeks)

    # add a line representing O3a / O3b divide,
    # and add text labels
    run_break = Span(
        location=mid, dimension="height", line_color="black", line_width=2
    )
    p.add_layout(run_break)
    p.add_layout(Label(x=mid - 3.5, y=15, text="O3a", text_font_size="20pt"))
    p.add_layout(Label(x=mid + 1, y=15, text="O3b", text_font_size="20pt"))

    legends = []
    for i, far in enumerate(fars):

        c = p.circle(x=weeks, y=original[:, i], size=10, color=palette[i])

        legends.append(LegendItem(renderers=[c], label=f"FAR {far}/month"))

    legend = Legend(items=legends, click_policy="mute")
    p.add_layout(legend, "right")

    save(p, filename="sensitive-volume-over-time.html")
    export_svg(p, filename="sensitive-volume-over-time.svg")


if __name__ == "__main__":
    main()
