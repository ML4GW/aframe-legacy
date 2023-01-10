from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import bilby

import logging

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Spinner
from bokeh.plotting import figure
from vizapp.priors import gaussian_masses

from bbhnet.analysis.vt import VolumeTimeIntegral

MPC3_TO_GPC3 = 1e-9


class VolumeTimeVsFAR:
    def __init__(
        self, height, width, source_prior: "bilby.core.prior.PriorDict"
    ):
        self.height = height
        self.width = width
        self.source_prior = source_prior
        self.keys = self.source_prior.keys()

        self.fars = np.logspace(0, 7, 7)
        self.configure_sources()
        self.configure_widgets()
        self.configure_plots()
        self.logger = logging.getLogger("VT vs FAR")
        self.logger.debug(self.fars)

    def configure_plots(
        self,
    ):

        self.figure = figure(
            height=self.height,
            width=self.width,
            x_axis_type="log",
            y_axis_type="log",
            x_axis_label="FAR (yr ^-1)",
            y_axis_label="<VT> Gpc^3 yr",
        )

        self.figure.line(
            x="far",
            y="vt",
            source=self.source,
            line_width=2,
            line_alpha=0.6,
            line_color="blue",
        )

        # Add error bars

        self.figure.segment(
            source=self.source,
            x0="far",
            y0="error_low",
            x1="far",
            y1="error_high",
            line_width=2,
        )

        self.layout = row(self.widgets, self.figure)

    def configure_widgets(self):
        self.m1_selector = Spinner(title="mass_1", value=30, low=5, high=60)
        self.m2_selector = Spinner(
            title="mass_2", value=29, low=5, high=self.m1_selector.value
        )
        self.m1_selector.on_change("value", self.calculate_vt)
        self.m2_selector.on_change("value", self.calculate_vt)

        self.widgets = column(self.m1_selector, self.m2_selector)

    def configure_sources(self):
        self.source = ColumnDataSource(
            data=dict(
                far=[],
                vt=[],
                error_low=[],
                error_high=[],
            )
        )

    def calculate_vt(self, attr, old, new):

        m1_mean = self.m1_selector.value
        m2_mean = self.m2_selector.value
        self.logger.debug(f"Calculating VT for m1 = {m1_mean}, m2 = {m2_mean}")
        target = gaussian_masses(m1_mean, m2_mean, sigma=3)

        fars = []
        vts = []
        uncertainties = []
        for far in self.fars:
            self.figure.title.text = (
                f"VT vs FAR for m1 = {m1_mean}, m2 = {m2_mean}"
            )
            # downselect to injections that are detected at this FAR
            indices = self.foreground.fars < far

            # parse foreground statistics into a dictionary
            # compatible with bilbys prior.prob method
            recovered_parameters = {
                "mass_1": self.foreground.m1s[indices],
                "mass_2": self.foreground.m2s[indices],
                "luminosity_distance": self.foreground.distances[indices],
                "ra": self.foreground.ras[indices],
                "dec": self.foreground.decs[indices],
            }

            volume_time_integral = VolumeTimeIntegral(
                source=self.source_prior,
                recovered_parameters=recovered_parameters,
                n_injections=self.n_injections,
                livetime=self.foreground.livetime,
            )

            vt, uncertainty = volume_time_integral.calculate_vt(target=target)

            # convert vt into Gpc^3 yr
            vt *= MPC3_TO_GPC3
            uncertainty *= MPC3_TO_GPC3
            fars.append(far)
            vts.append(vt)
            uncertainties.append(uncertainty)

        vts = np.array(vts)
        uncertainties = np.array(uncertainties)
        self.logger.debug(uncertainties)
        self.source.data = {
            "far": fars,
            "vt": vts,
            "error_low": vts - uncertainties,
            "error_high": vts + uncertainties,
        }

    def update(self, foreground):
        self.foreground = foreground
        self.n_injections = len(foreground.injection_times)
        self.livetime = foreground.livetime
        self.calculate_vt(None, None, None)
