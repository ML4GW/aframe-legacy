from vizapp.pages.analysis.background import DistributionPlot
from vizapp.pages.page import Page


class AnalysisPage(Page):
    def __init__(self, app):
        self.app = app
        self.distribution_plot = DistributionPlot(self)
        self.initialize_sources()

    def initialize_sources(self) -> None:
        self.distribution_plot.initialize_sources()

    def get_layout(self):
        return self.distribution_plot.get_layout(height=400, width=1000)

    def update(self):
        self.distribution_plot.update()
