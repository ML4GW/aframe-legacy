from vizapp.pages.analysis.background import BackgroundPlot
from vizapp.pages.page import Page


class AnalysisPage(Page):
    def __init__(self, app):
        super().__init__(app)
        self.background_plot = BackgroundPlot(self)
