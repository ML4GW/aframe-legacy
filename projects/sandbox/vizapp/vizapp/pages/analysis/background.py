import numpy as np
from bokeh.models import (
    BoxSelectTool,
    ColumnDataSource,
    HoverTool,
    LogAxis,
    Range1d,
)
from bokeh.plotting import figure
from vizapp import palette


class DistributionPlot:
    def __init__(self, page) -> None:
        self.page = page

        self.bckgd_color = palette[4]
        self.frgd_color = palette[2]

    def asdict(self, background, foreground):
        fore_attrs = [
            "chirp_mass",
            "shift",
            "mass_1",
            "mass_2",
            "snr",
            "detection_statistic",
            "shift",
            "time",
        ]
        back_attrs = ["detection_statistic"]
        background = {attr: getattr(background, attr) for attr in back_attrs}
        foreground = {attr: getattr(foreground, attr) for attr in fore_attrs}
        return background, foreground

    def initialize_sources(self):
        self.bar_source = ColumnDataSource(dict(center=[], top=[], width=[]))
        self.background_source = ColumnDataSource(dict())
        self.foreground_source = ColumnDataSource(dict())

    def get_layout(self, height, width):
        self.distribution_plot = figure(
            height=height,
            width=width,
            y_axis_type="log",
            x_axis_label="Detection statistic",
            y_axis_label="Background survival function",
            y_range=(0, 1),  # set dummy values to allow updating later
            tools="box_zoom,reset",
        )
        self.distribution_plot.yaxis.axis_label_text_color = self.bckgd_color

        # add box select tool for selecting ranges
        # of background events to further analyze
        box_select = BoxSelectTool(dimensions="width")
        self.distribution_plot.add_tools(box_select)
        self.distribution_plot.toolbar.active_drag = box_select
        # self.bar_source.selected.on_change("indices", self.update_background)

        self.distribution_plot.extra_y_ranges = {"SNR": Range1d(1, 10)}
        axis = LogAxis(
            axis_label="Injected Event SNR",
            axis_label_text_color=self.frgd_color,
            y_range_name="SNR",
        )
        self.distribution_plot.add_layout(axis, "right")

        self.plot_data()
        return self.distribution_plot

    def plot_data(self):
        injection_renderer = self.distribution_plot.circle(
            x="detection_statistic",
            y="snr",
            fill_color=self.frgd_color,
            line_color=self.frgd_color,
            line_width=0.5,
            fill_alpha=0.2,
            line_alpha=0.4,
            selection_fill_alpha=0.2,
            selection_line_alpha=0.3,
            nonselection_fill_alpha=0.2,
            nonselection_line_alpha=0.3,
            y_range_name="SNR",
            source=self.foreground_source,
        )

        # add hover tool for analyzing additional attributes
        hover = HoverTool(
            tooltips=[
                ("Hanford GPS time", "@{time}{0.000}"),
                ("Shifts", "@shift"),
                ("SNR", "@snr"),
                ("Detection statistic", "@{detection_statistic}"),
                ("Mass 1", "@{mass_1}"),
                ("Mass 2", "@{mass_2}"),
                ("Chirp Mass", "@{chirp_mass}"),
            ],
            renderers=[injection_renderer],
        )
        self.distribution_plot.add_tools(hover)

        self.distribution_plot.vbar(
            "center",
            top="top",
            bottom=0.1,
            width="width",
            fill_color=self.bckgd_color,
            line_color="#000000",
            fill_alpha=0.4,
            line_alpha=0.6,
            line_width=0.5,
            selection_fill_alpha=0.6,
            selection_line_alpha=0.8,
            nonselection_fill_alpha=0.2,
            nonselection_line_alpha=0.3,
            source=self.bar_source,
        )

    def update(self):
        # apply vetoes to background, update data source, and update title
        background = self.page.app.background
        foreground = self.page.app.foreground

        background = background[~self.page.app.veto_mask]

        title = (
            "{} background events from {:0.2f} "
            "days worth of data; {} injections overlayed"
        ).format(
            len(background),
            background.Tb / 3600 / 24,
            len(foreground),
        )
        background_dict, foreground_dict = self.asdict(background, foreground)

        self.background_source.data = background_dict
        self.foreground_source.data = foreground_dict

        self.distribution_plot.title.text = title

        # update bar plot
        hist, bins = np.histogram(background.detection_statistic, bins=100)
        hist = np.cumsum(hist[::-1])[::-1]
        self.distribution_plot.y_range.start = 0.1
        self.distribution_plot.y_range.end = 2 * hist.max()

        self.bar_source.data.update(
            center=(bins[:-1] + bins[1:]) / 2,
            top=hist,
            width=0.95 * (bins[1:] - bins[:-1]),
        )

        # update snr axis of plot
        # add extra y axis range to show SNR's of events
        self.distribution_plot.extra_y_ranges["SNR"].start = (
            0.5 * foreground.snr.min()
        )
        self.distribution_plot.extra_y_ranges["SNR"].end = (
            2 * foreground.snr.max()
        )
