from bokeh.layouts import column
from vizapp.pages.analysis.background import DistributionPlot
from vizapp.pages.analysis.event_inspector import EventInspectorPlot
from vizapp.pages.page import Page


class AnalysisPage(Page):
    def __init__(self, app):
        self.app = app
        self.event_inspector = EventInspectorPlot(self)
        self.distribution_plot = DistributionPlot(self, self.event_inspector)

        self.initialize_sources()

    def initialize_sources(self) -> None:
        self.distribution_plot.initialize_sources()
        self.event_inspector.initialize_sources()

    def get_layout(self):
        event_inspector = self.event_inspector.get_layout(
            height=400, width=600
        )
        distribution = self.distribution_plot.get_layout(
            height=400, width=1500
        )
        return column(distribution, event_inspector)

    def update(self):
        self.distribution_plot.update()
