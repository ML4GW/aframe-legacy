from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

if TYPE_CHECKING:
    from astropy.cosmology import Cosmology

import astropy.units as u
import h5py
import numpy as np
from astropy.cosmology import z_at_value
from bilby.core.prior import Interped, PriorDict
from bilby.gw.conversion import generate_mass_parameters


def chirp_mass(m1, m2):
    return ((m1 * m2) ** 3 / (m1 + m2)) ** (1 / 5)


def luminosity_distance_to_chirp_distance(d, mchirp):
    fiducial = 1.4 / 2 ** (1 / 5)
    factor = (fiducial / mchirp) ** (5 / 6)
    return d * factor


def chirp_distance_to_luminosity_distance(dchirp, mchirp):
    fiducial = 1.4 / 2 ** (1 / 5)
    factor = (fiducial / mchirp) ** (5 / 6)
    return dchirp / factor


def mass_condition_uniform(reference_params, mass_1):
    return dict(
        minimum=reference_params["minimum"],
        maximum=mass_1,
    )


def mass_condition_powerlaw(reference_params, mass_1):
    return dict(
        alpha=reference_params["alpha"],
        minimum=reference_params["minimum"],
        maximum=mass_1,
    )


def mass_constraints(samples):
    if "mass_1" not in samples or "mass_2" not in samples:
        raise KeyError("mass_1 and mass_1 must exist to have a mass_ratio")
    samples["mass_ratio"] = samples["mass_2"] / samples["mass_1"]
    samples["chirp_mass"] = chirp_mass(samples["mass_1"], samples["mass_2"])
    return samples


def transpose(d: Dict[str, List]):
    return [dict(zip(d, col)) for col in zip(*d.values())]


def convert_distance_to_redshift(
    samples: Dict[str, np.ndarray], cosmology: "Cosmology"
):
    """
    If redshift is not among the keys in `samples`, we assume that we
    have sampled either luminosity distance or chirp distance.
    Luminosity distance is assumed to be in units of Mpc
    """
    if "luminosity_distance" not in samples:
        samples["luminosity_distance"] = chirp_distance_to_luminosity_distance(
            samples["chirp_distance"], samples["chirp_mass"]
        )
    samples["redshift"] = z_at_value(
        cosmology.luminosity_distance, samples["luminosity_distance"] * u.Mpc
    )

    return samples


def _generate_detector_frame_masses(samples: Dict[str, np.ndarray]):
    mass_keys = ["mass_1", "mass_2", "chirp_mass", "total_mass"]
    for key in mass_keys:
        if key in samples:
            samples[f"{key}_source"] = samples[key]
            samples[key] *= 1 + samples["redshift"]

    return samples


def _generate_source_frame_masses(samples: Dict[str, np.ndarray]):
    mass_keys = ["mass_1", "mass_2", "chirp_mass", "total_mass"]
    for key in mass_keys:
        if key in samples:
            samples[f"{key}_source"] = samples[key] / (1 + samples["redshift"])

    return samples


def parameter_conversion(
    samples: Dict[str, np.ndarray], cosmology: "Cosmology", source_frame: bool
):
    """
    Convert mass parameters and distance parameters into various useful forms.
    Requires 2 mass parameters and one distance parameter in `samples`.
    Assumes that the given masses are specified in the detector frame.
    """

    samples = generate_mass_parameters(samples)
    if "redshift" not in samples:
        samples = convert_distance_to_redshift(samples, cosmology)
    if source_frame:
        samples = _generate_detector_frame_masses(samples)
    else:
        samples = _generate_source_frame_masses(samples)

    fiducial = 1.4 / (2 ** (1 / 5))
    factor = (fiducial / samples["chirp_mass"]) ** (5 / 6)

    # We'll assume that we don't have both chirp and luminosity distance
    # We know that we have redshift at the very least
    if "chirp_distance" in samples:
        samples["luminosity_distance"] = samples["chirp_distance"] / factor
    elif "luminosity_distance" in samples:
        samples["chirp_distance"] = samples["luminosity_distance"] * factor
    else:
        samples["luminosity_distance"] = cosmology.luminosity_distance(
            samples["redshift"]
        ).value
        samples["chirp_distance"] = samples["luminosity_distance"] * factor

    return samples


def pdf_from_events(
    param_values: Sequence[float],
    grid_size: int = 100,
    spacing: str = "lin",
) -> Tuple[Sequence[float], Sequence[float]]:
    """
    Estimates the probability distribution of a parameter based on
    a list of sampled values. Currently does this by just creating
    a histogram of the values, but might consider doing a KDE in
    the future

    Args:
        param_values:
            A list of parameter values drawn from the distribution
            to be estimated
        grid_size:
            The number of points at which to estimate the pdf
        spacing:
            The spacing type of the grid, either linear or log

    Returns:
        grid:
            The values at which the pdf was estimated
        pdf:
            The estimated pdf
    """
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


def read_priors_from_file(event_file: Path, *parameters: str) -> PriorDict:
    """
    Reads in a file containing sets of GW parameters and
    returns a set of interpolated priors
    The expected structure is based off the event file from
    here: https://dcc.ligo.org/T2100512

    Args:
        event_file: An hdf5 file containing event parameters
        parameters: Optional, a list of parameters to read from the file

    Returns:
        prior: A PriorDict with priors based on the event file
    """
    prior = PriorDict()
    with h5py.File(event_file, "r") as f:
        events = f["events"]
        field_names = parameters or events.dtype.names
        for name in field_names:
            grid, pdf = pdf_from_events(events[name])
            prior[name] = Interped(grid, pdf, np.min(grid), np.max(grid))

    return prior
