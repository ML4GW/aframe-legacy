from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from bilby.core.prior import (
    Constraint,
    Cosine,
    Interped,
    PowerLaw,
    Sine,
    Uniform,
)
from bilby.gw.prior import (
    BBHPriorDict,
    UniformComovingVolume,
    UniformSourceFrame,
)

# Unit names
msun = r"$M_{\odot}$"
mpc = "Mpc"
rad = "rad"


def pdf_from_events(
    param_values: Sequence[float],
    grid_size: int = 100,
    spacing: str = "lin",
):
    param_min = np.min(param_values)
    param_max = np.max(param_values)
    if spacing == "lin":
        bins = np.linspace(param_min, param_max, grid_size + 1)
        grid = (bins[:-1] + bins[1:]) / 2
    elif spacing == "log":
        min_exp = np.log10(param_min)
        max_exp = np.log10(param_max)
        bins = np.logspace(min_exp, max_exp, grid_size + 1)
        grid = np.sqrt(bins[:-1] * bins[1:])
    else:
        raise ValueError("Spacing must be either 'lin' or 'log'")

    pdf, _ = np.histogram(param_values, bins, density=True)

    return grid, pdf


def read_priors_from_file(event_file: Path, *parameters: str) -> BBHPriorDict:
    prior = BBHPriorDict()
    with h5py.File(event_file, "r") as f:
        events = f["events"]
        events = events[(events["mass_1"] > 5) & (events["mass_2"] > 5)]
        field_names = parameters or events.dtype.names
        for name in field_names:
            # We'll specify mass_2 via mass_ratio at sampling time
            if name == "mass_2":
                continue
            grid, pdf = pdf_from_events(events[name])
            prior[name] = Interped(grid, pdf, np.min(grid), np.max(grid))

    return prior


def assign_prior_names(prior: BBHPriorDict) -> BBHPriorDict:
    for key in prior.keys():
        if hasattr(prior[key], "name"):
            print("check")
            prior[key].name = key
    return prior


def uniform_extrinsic():
    prior = BBHPriorDict()
    prior["dec"] = Cosine()
    prior["ra"] = Uniform(0, 2 * np.pi)
    prior["theta_jn"] = 0
    prior["phase"] = 0
    prior = assign_prior_names(prior)

    return prior


def nonspin_bbh():
    prior = uniform_extrinsic()
    prior["mass_1"] = Uniform(5, 100, unit=msun)
    prior["mass_2"] = Uniform(5, 100, unit=msun)
    prior["mass_ratio"] = Constraint(0.2, 5)
    prior["luminosity_distance"] = UniformSourceFrame(
        100, 3000, unit=mpc, name="luminosity_distance"
    )
    prior["psi"] = 0
    prior["a_1"] = 0
    prior["a_2"] = 0
    prior["tilt_1"] = 0
    prior["tilt_2"] = 0
    prior["phi_12"] = 0
    prior["phi_jl"] = 0
    prior = assign_prior_names(prior)

    return prior


def end_o3_ratesandpops():
    prior = uniform_extrinsic()
    prior["mass_1"] = PowerLaw(alpha=-2.35, minimum=2, maximum=100, unit=msun)
    prior["mass_2"] = PowerLaw(alpha=1, minimum=2, maximum=100, unit=msun)
    prior["mass_ratio"] = Constraint(0.02, 1)
    prior["luminosity_distance"] = UniformComovingVolume(
        100, 15000, unit=mpc, name="luminosity_distance"
    )
    prior["psi"] = 0
    prior["a_1"] = Uniform(0, 0.998)
    prior["a_2"] = Uniform(0, 0.998)
    prior["tilt_1"] = Sine(unit=rad)
    prior["tilt_2"] = Sine(unit=rad)
    prior["phi_12"] = Uniform(0, 2 * np.pi)
    prior["phi_jl"] = 0
    prior = assign_prior_names(prior)

    return prior


def power_law_dip_break():
    prior = uniform_extrinsic()
    event_file = "/home/william.benoit/\
        O1O2O3all_mass_h_iid_mag_iid_tilt_powerlaw_redshift_maxP_events_all.h5"
    prior |= read_priors_from_file(event_file)
    prior = assign_prior_names(prior)

    return prior
