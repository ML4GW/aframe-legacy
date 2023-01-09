import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from astropy import cosmology
from astropy import units as u

if TYPE_CHECKING:
    from astropy.cosmology import Cosmology
    import bilby

import numpy as np
from scipy.integrate import quad

PI_OVER_TWO = math.pi / 2


def calculate_astrophysical_volume(
    cosmology: "Cosmology",
    dl_min: float,
    dl_max: float,
    dec_min: float = -PI_OVER_TWO,
    dec_max: float = PI_OVER_TWO,
):
    """
    Calculates the astrophysical volume over which injections have been made.
    See equation 4) in https://arxiv.org/pdf/1712.00482.pdf

    Args:
        dl_min: minimum distance of injections in Mpc
        dl_max: maximum distance of injections in Mpc
        dec_min: minimum declination of injections in radians
        dec_max: maximum declination of injections in radians
        cosmology: astropy cosmology object

    Returns volume in Mpc^3
    """

    # given passed cosmology, calculate the comoving distance
    # at the minimum and maximum distance of injections
    zmin, zmax = cosmology.z_at_value(
        cosmology.comoving_distance, [dl_min, dl_max]
    )

    # calculate the angular volume of the sky
    # over which injections have been made
    dec_min, dec_max = -np.pi / 2 * u.rad, np.pi / 2 * u.rad
    theta_max, theta_min = (
        np.pi / 2 - dec_min.to("rad").value,
        np.pi / 2 - dec_max.to("rad").value,
    )
    omega = -2 * math.pi * (np.cos(theta_max) - np.cos(theta_min))

    # calculate the volume of the universe
    # over which injections have been made
    integrand = (
        lambda z: 1.0
        / (1 + z)
        * (cosmology.differential_comoving_volume(z)).value
    )
    volume, _ = quad(integrand, zmin, zmax) * u.Mpc**3 * omega
    return volume


@dataclass
class VolumeTimeIntegral:
    """
    Class for calculating VT metrics using importance sampling.

    Args:
        source:
            Bilby PriorDict of the source distribution
            used to create injections
        recovered_parameters:
            Array of recovered injections
        n_injections:
            Number of total injections
        livetime:
            Livetime in seconds over which injections were performed
        cosmology:
            Astropy Cosmology object used for volume calculation
    """

    source: "bilby.core.prior.PriorDict"
    recovered_parameters: np.ndarray
    n_injections: int
    livetime: float
    cosmology: "Cosmology"

    def __post_init__(self):
        # calculate the astrophysical volume over
        # which injections have been made.
        self.volume = calculate_astrophysical_volume(
            dl_min=self.source["luminosity_distance"].minimum,
            dl_max=self.source["luminosity_distance"].maximum,
            dec_min=self.source["dec"].minimum,
            dec_max=self.source["dec"].maximum,
            cosmology=cosmology,
        )

    def weights(self, target: Optional["bilby.core.prior.PriorDict"] = None):
        """
        Calculate the weights for the samples.
        """

        # if no target distribution is passed,
        # use the source distribution
        if target is None:
            target = self.source

        weights = target.prob(self.recovered_parameters) / self.source.prob(
            self.recovered_parameters
        )
        return weights

    def calculate_vt(
        self,
        target: Optional["bilby.core.prior.PriorDict"] = None,
    ):
        """
        Calculates the VT and its uncertainty.

        Args:
            target:
                Bilby PriorDict of the target distribution
                used for importance sampling. If None, the source
                distribution is used.
        """
        weights = self.weights(target) ** self.volume * self.livetime
        vt = np.sum(weights) / self.n_injections
        uncertainty = (np.sum(weights**2) / self.n_injections) - (
            vt**2 / self.n_injections
        )
        return vt, uncertainty
