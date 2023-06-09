from bokeh.models import BoxSelectTool, ColumnDataSource
from bokeh.plotting import figure
from vizapp import palette


class BackgroundPlot:
    def __init__(self, page) -> None:
        self.page = page
        self.background = page.app.background
        self.foreground = page.app.foreground

    def initialize_sources(self):
        self.bar_source = ColumnDataSource(dict(center=[], top=[], width=[]))
        self.foreground_source = ColumnDataSource(
            dict(
                injection_time=[],
                detection_statistic=[],
                event_time=[],
                shift=[],
                snr=[],
                chirp_mass=[],
                distances=[],
                mass_1=[],
                mass_2=[],
                size=[],
            )
        )

        self.background_source = ColumnDataSource(
            dict(
                x=[],
                event_time=[],
                detection_statistic=[],
                color=[],
                label=[],
                count=[],
                shift=[],
                size=[],
            )
        )

        def update(self):
            pass

        def get_layout(self, height, width):
            bckgd_color = palette[4]
            # frgd_color = palette[2]

            self.distribution_plot = figure(
                height=height,
                width=int(width * 0.55),
                y_axis_type="log",
                x_axis_label="Detection statistic",
                y_axis_label="Background survival function",
                y_range=(0, 1),  # set dummy values to allow updating later
                tools="box_zoom,reset",
            )
            self.distribution_plot.yaxis.axis_label_text_color = bckgd_color

            box_select = BoxSelectTool(dimensions="width")
            self.distribution_plot.add_tools(box_select)
            self.distribution_plot.toolbar.active_drag = box_select
