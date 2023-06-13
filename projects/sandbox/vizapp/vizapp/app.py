import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

import bilby
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Div, MultiChoice, TabPanel, Tabs
from vizapp.pages import AnalysisPage, PerformanceSummaryPage

from aframe.analysis.ledger.events import EventSet, RecoveredInjectionSet
from aframe.analysis.ledger.injections import InjectionParameterSet

if TYPE_CHECKING:
    import torch
    from astropy.cosmology import Cosmology
    from vizapp.vetoes import VetoParser

    from .structures import Preprocessor


class VizApp:
    def __init__(
        self,
        model: "torch.nn.Module",
        preprocessor: "Preprocessor",
        base_directory: Path,
        data_directory: Path,
        cosmology: "Cosmology",
        source_prior: "bilby.core.prior.PriorDict",
        ifos: List[str],
        sample_rate: float,
        inference_stride: int,
        kernel_length: float,
        fduration: float,
        valid_frac: float,
        veto_parser: "VetoParser",
    ) -> None:
        self.logger = logging.getLogger("vizapp")
        self.logger.debug("Loading analyzed distributions")

        # set a bunch of attributes
        self.model = model
        self.preprocessor = preprocessor
        self.veto_parser = veto_parser
        self.ifos = ifos
        self.source_prior = source_prior
        self.cosmology = cosmology
        self.sample_rate = sample_rate
        self.fduration = fduration
        self.valid_frac = valid_frac
        self.inference_stride = inference_stride
        self.strain_dir = data_directory / "test" / "background"
        self.kernel_length = kernel_length
        self.background_length = kernel_length - (fduration + 1)

        # load results and data from the run we're visualizing
        self.response_path = data_directory / "test" / "waveforms.h5"
        rejected = data_directory / "test" / "rejected-parameters.h5"

        infer_dir = base_directory / "infer"
        self.background = EventSet.read(infer_dir / "background.h5")
        self.foreground = RecoveredInjectionSet.read(
            infer_dir / "foreground.h5"
        )
        self.rejected_params = InjectionParameterSet.read(rejected)

        # set up our veto selecter and set up the initially
        # blank veto mask, use this to update the sources
        # for all our pages
        self.veto_selecter = self.get_veto_selecter()
        self.veto_selecter.on_change("value", self.update_vetos)

        # initialize all our pages and their constituent plots
        self.pages, tabs = [], []
        for page in [PerformanceSummaryPage, AnalysisPage]:
            page = page(self)
            self.pages.append(page)

            title = page.__class__.__name__.replace("Page", "")
            tab = TabPanel(child=page.get_layout(), title=title)
            tabs.append(tab)

        # once pages and their sources are intialized, update vetoes
        self.update_vetos(None, None, [])

        # set up a header with a title and the selecter
        title = Div(text="<h1>aframe Performance Dashboard</h1>", width=500)
        header = row(title, self.veto_selecter)

        # generate the final layout
        tabs = Tabs(tabs=tabs)
        self.layout = column(header, tabs)
        self.logger.info("Application ready!")

    def get_veto_selecter(self):
        options = ["CAT1", "CAT2", "CAT3", "GATES"]
        self.vetoes = {}
        for label in options:
            vetos = self.veto_parser.get_vetoes(label)
            veto_mask = np.zeros(len(self.background), dtype=bool)
            for ifo in self.ifos:
                segments = vetos[ifo]

                if len(segments) != 0:
                    mask = segments[:, :1] < self.background.time
                    mask &= segments[:, 1:] > self.background.time

                    # mark a background event as vetoed
                    # if it falls into _any_ of the segments
                    veto_mask |= mask.any(axis=0)
            self.vetoes[label] = veto_mask

        self.veto_mask = np.zeros(len(self.background), dtype=bool)
        return MultiChoice(title="Applied Vetoes", value=[], options=options)

    def update_vetos(self, attr, old, new):
        if not new:
            # no vetoes selected, so mark all background
            # events as not-vetoed
            self.veto_mask = np.zeros_like(self.veto_mask, dtype=bool)
        else:
            # mark a background event as vetoed if any
            # of the currently selected labels veto it
            mask = False
            for label in new:
                mask |= self.vetoes[label]
            self.veto_mask = mask
        # now update all our pages to factor
        # in the vetoed data
        for page in self.pages:
            page.update()

    def __call__(self, doc):
        doc.add_root(self.layout)
