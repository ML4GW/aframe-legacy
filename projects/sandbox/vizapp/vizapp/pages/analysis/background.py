from dataclasses import asdict

import numpy as np
from bokeh.models import BoxSelectTool, ColumnDataSource
from bokeh.plotting import figure
from vizapp import palette


class BackgroundPlot:
    def __init__(self, page) -> None:
        self.page = page
        self.background = page.app.background
        self.foreground = page.app.foreground

        self.bckgd_color = palette[4]
        self.frgd_color = palette[2]

    def asdict(self, background, foreground):
        # cast as dicts and remove metadata that is
        # incompatible with ColumnDataSource
        # TODO: add asdict methods to ledger api?
        background = asdict(background)
        background.pop("Tb")

        foreground = asdict(foreground)
        [
            foreground.pop(key)
            for key in ["duration", "sample_rate", "num_injections", "Tb"]
        ]
        return background, foreground

    def initialize_sources(self):
        self.bar_source = ColumnDataSource(dict(center=[], top=[], width=[]))

        background, foreground = self.asdict(
            self.page.app.background, self.page.app.foreground
        )
        self.background_source = ColumnDataSource(background)
        self.foreground_source = ColumnDataSource(foreground)

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

        box_select = BoxSelectTool(dimensions="width")
        self.distribution_plot.add_tools(box_select)
        self.distribution_plot.toolbar.active_drag = box_select

        self.plot_data()
        return self.distribution_plot

    def plot_data(self):
        self.distribution_plot.circle(
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
            source=self.foreground_source,
        )

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
            self.background.Tb / 3600 / 24,
            len(foreground),
        )
        background, foreground = self.asdict(background, foreground)

        self.foreground_source.data = foreground
        self.background_source.data = background

        self.distribution_plot.title.text = title

        # update bar plot
        hist, bins = np.histogram(
            self.background.detection_statistic, bins=100
        )
        hist = np.cumsum(hist[::-1])[::-1]
        self.distribution_plot.y_range.start = 0.1
        self.distribution_plot.y_range.end = 2 * hist.max()

        self.bar_source.data.update(
            center=(bins[:-1] + bins[1:]) / 2,
            top=hist,
            width=0.95 * (bins[1:] - bins[:-1]),
        )
