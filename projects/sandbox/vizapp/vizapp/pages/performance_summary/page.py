from vizapp.pages.page import Page
from vizapp.pages.performance_summary.sensitive_volume import (
    SensitiveVolumePlot,
)


class PerformanceSummaryPage(Page):
    def __init__(self, app):
        self.app = app
        self.sensitive_volume = SensitiveVolumePlot(self)
        self.initialize_sources()

    def initialize_sources(self) -> None:
        self.sensitive_volume.initialize_sources()

    def get_layout(self):
        return self.sensitive_volume.get_layout(height=400, width=1000)

    def update(self):
        self.sensitive_volume.update()
