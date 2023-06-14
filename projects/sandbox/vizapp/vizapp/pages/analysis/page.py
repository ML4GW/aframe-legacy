from bokeh.layouts import column
from vizapp.pages.analysis.background import DistributionPlot
from vizapp.pages.analysis.event_inspector import (
    EventAnalyzer,
    EventInspectorPlot,
)
from vizapp.pages.page import Page


class AnalysisPage(Page):
    def __init__(self, app):
        self.app = app
        analyzer = self.get_analyzer()

        self.event_inspector = EventInspectorPlot(self, analyzer)
        self.distribution_plot = DistributionPlot(self, self.event_inspector)

        self.initialize_sources()

    def get_analyzer(self):
        return EventAnalyzer(
            self.app.model,
            self.app.preprocessor,
            self.app.strain_dir,
            self.app.qscan_dir,
            self.app.response_path,
            self.app.fduration,
            self.app.kernel_length,
            self.app.inference_sampling_rate,
            self.app.ifos,
            self.app.background_length,
            self.app.sample_rate,
            self.app.padding,
            self.app.integration_length,
        )

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
