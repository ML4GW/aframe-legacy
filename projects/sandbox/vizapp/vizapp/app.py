import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

import h5py
from bokeh.layouts import column, row
from bokeh.models import Div, MultiChoice, Panel, Select, Tabs
from vizapp.distributions import get_foreground, load_results
from vizapp.plots import BackgroundPlot, EventInspectorPlot, PerfSummaryPlot

if TYPE_CHECKING:
    from vizapp.vetoes import VetoeParser


class VizApp:
    def __init__(
        self,
        timeslides_results_dir: Path,
        timeslides_strain_dir: Path,
        train_data_dir: Path,
        vetoe_parser: "VetoeParser",
        ifos: List[str],
        sample_rate: float,
        fduration: float,
        valid_frac: float,
    ) -> None:
        self.logger = logging.getLogger("vizapp")
        self.logger.debug("Loading analyzed distributions")
        self.vetoe_parser = vetoe_parser
        self.ifos = ifos

        # load in foreground and background distributions
        self.distributions = load_results(timeslides_results_dir)

        # create version with vetoes
        self.vetoed_distributions = load_results(timeslides_results_dir)

        self.logger.debug("Structuring distribution events")
        self.foregrounds = {}
        for norm, results in self.distributions.items():

            foreground = get_foreground(
                results, timeslides_strain_dir, timeslides_results_dir, norm
            )
            self.foregrounds[norm] = foreground

        self.logger.debug("Configuring widgets")
        self.configure_widgets()
        self.logger.debug("Configuring plots")
        self.configure_plots(
            sample_rate,
            fduration,
            1 - valid_frac,
            train_data_dir,
            timeslides_strain_dir,
            timeslides_results_dir,
        )
        self.update_norm(None, None, self.norm_select.options[0])

        self.logger.info("Application ready!")

    def configure_widgets(self):
        header = Div(text="<h1>BBHNet Performance Dashboard</h1>", width=500)

        norm_options = list(self.distributions)
        if None in norm_options:
            value = None
            options = [None] + sorted([i for i in norm_options if i])
        else:
            options = sorted(norm_options)
            value = options[0]

        self.norm_select = Select(
            title="Normalization period [s]",
            value=str(value),
            options=list(map(str, options)),
        )
        self.norm_select.on_change("value", self.update_norm)

        self.vetoe_labels = ["CAT1", "CAT2", "CAT3", "GATES"]
        self.vetoe_choices = MultiChoice(value=[], options=self.vetoe_labels)
        self.vetoe_choices.on_change("value", self.update_vetoes)

        self.widgets = row(header, self.norm_select, self.vetoe_choices)

    def configure_plots(
        self,
        sample_rate,
        fduration,
        train_frac,
        train_data_dir,
        timeslides_strain_dir,
        timeslides_results_dir,
    ):
        self.perf_summary_plot = PerfSummaryPlot(300, 800)

        backgrounds = {}
        for ifo in self.ifos:
            with h5py.File(train_data_dir / f"{ifo}_background.h5", "r") as f:
                bkgd = f["hoft"][:]
                bkgd = bkgd[: int(train_frac * len(bkgd))]
                backgrounds[ifo] = bkgd

        self.event_inspector = EventInspectorPlot(
            height=300,
            width=1500,
            response_dir=timeslides_results_dir,
            strain_dir=timeslides_strain_dir,
            fduration=fduration,
            sample_rate=sample_rate,
            freq_low=30,
            freq_high=300,
            **backgrounds,
        )

        self.background_plot = BackgroundPlot(300, 1200, self.event_inspector)

        summary_tab = Panel(
            child=self.perf_summary_plot.layout, title="Summary"
        )

        analysis_layout = column(
            self.background_plot.layout, self.event_inspector.layout
        )
        analysis_tab = Panel(child=analysis_layout, title="Analysis")
        tabs = Tabs(tabs=[summary_tab, analysis_tab])
        self.layout = column(self.widgets, tabs)

    def update_norm(self, attr, old, new):
        norm = None if new == "None" else float(new)

        self.logger.debug(f"Updating plots with normalization value {norm}")
        foreground = self.foregrounds[norm]
        background = self.vetoed_distributions[norm].background

        self.perf_summary_plot.update(foreground)
        self.background_plot.update(foreground, background, norm)

    def update_vetoes(self, attr, old, new):

        # apply requested vetoes for each norm window
        if len(new) > 0:

            self.logger.debug(f"Active vetoes: {new}")
            for norm, results in self.distributions.items():
                background = results.background.copy()
                self.logger.debug(
                    f"{len(background.events)} events for norm {norm} pre veto"
                )
                for category in new:
                    vetoes = self.vetoe_parser.get_vetoes(category)
                    background.apply_vetoes(**vetoes)
                self.logger.debug(
                    f"{len(background.events)} events "
                    f"for norm {norm} post veto"
                )

                self.vetoed_distributions[norm].background = background
                # update fars of foreground based on new background
                foreground = self.foregrounds[norm]
                foreground.fars = background.far(
                    foreground.detection_statistics
                )
                self.foregrounds[norm] = foreground
        else:
            # remove vetoes by referring back to non-vetoe distributions
            self.logger.debug("Removing all vetoes")
            for norm, results in self.distributions.items():
                background = results.background
                self.vetoed_distributions[norm].background = background
                foreground = self.foregrounds[norm]
                foreground.fars = background.far(
                    foreground.detection_statistics
                )
                self.foregrounds[norm] = foreground

        # now that vetoes have been applied for all norms,
        # select the foreground and background for the currently selected norm
        current_norm = self.norm_select.value
        foreground = self.foregrounds[float(current_norm)]
        background = self.vetoed_distributions[float(current_norm)].background

        # update plots with
        self.logger.debug(
            "Updating plots with new distributions after changing vetoes"
        )

        self.perf_summary_plot.update(foreground)
        self.background_plot.update(foreground, background, current_norm)
        self.event_inspector.reset()

    def __call__(self, doc):
        doc.add_root(self.layout)
