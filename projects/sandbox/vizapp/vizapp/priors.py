import numpy as np
from bilby.core.prior import (
    ConditionalPowerLaw,
    ConditionalPriorDict,
    ConditionalTruncatedGaussian,
    Constraint,
    Cosine,
    Gaussian,
    PowerLaw,
    PriorDict,
    Sine,
    Uniform,
)
from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame

"""
This is a duplicate of the priors.py in the datagen project.

This is a temporary solution to avoid having to install the datagen
project which carries a torch dependency. We should probably refactor
the priors module into its own library again.
"""


def nonspin_bbh():

    prior_dict = PriorDict()
    prior_dict["mass_1"] = Uniform(
        name="mass_1", minimum=5, maximum=100, unit=r"$M_{\odot}$"
    )
    prior_dict["mass_2"] = Uniform(
        name="mass_2", minimum=5, maximum=100, unit=r"$M_{\odot}$"
    )
    prior_dict["mass_ratio"] = Constraint(
        name="mass_ratio", minimum=0.2, maximum=5.0
    )
    prior_dict["luminosity_distance"] = UniformSourceFrame(
        name="luminosity_distance", minimum=100, maximum=3000, unit="Mpc"
    )
    prior_dict["dec"] = Cosine(name="dec")
    prior_dict["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )
    prior_dict["theta_jn"] = 0
    prior_dict["psi"] = 0
    prior_dict["phase"] = 0
    prior_dict["a_1"] = 0
    prior_dict["a_2"] = 0
    prior_dict["tilt_1"] = 0
    prior_dict["tilt_2"] = 0
    prior_dict["phi_12"] = 0
    prior_dict["phi_jl"] = 0

    return prior_dict


def end_o3_ratesandpops():

    prior_dict = ConditionalPriorDict()
    prior_dict["mass_1"] = PowerLaw(
        name="mass_1", alpha=-2.35, minimum=2, maximum=100, unit=r"$M_{\odot}"
    )

    def condition_func(reference_params, mass_1):
        return dict(
            alpha=reference_params["alpha"],
            minimum=reference_params["minimum"],
            maximum=mass_1,
        )

    prior_dict["mass_2"] = ConditionalPowerLaw(
        name="mass_2",
        condition_func=condition_func,
        alpha=1,
        minimum=2,
        maximum=100,
        unit=r"$M_{\odot}",
    )
    prior_dict["luminosity_distance"] = UniformComovingVolume(
        name="luminosity_distance", minimum=100, maximum=15000, unit="Mpc"
    )
    prior_dict["dec"] = Cosine(name="dec")
    prior_dict["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )
    prior_dict["theta_jn"] = 0
    prior_dict["psi"] = 0
    prior_dict["phase"] = 0
    prior_dict["a_1"] = Uniform(name="a_1", minimum=0, maximum=0.998)
    prior_dict["a_2"] = Uniform(name="a_2", minimum=0, maximum=0.998)
    prior_dict["tilt_1"] = Sine(name="tilt_1", unit="rad")
    prior_dict["tilt_2"] = Sine(name="tilt_2", unit="rad")
    prior_dict["phi_12"] = Uniform(
        name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )
    prior_dict["phi_jl"] = 0

    return prior_dict


def gaussian_masses(m1: float, m2: float, sigma: float = 2):
    """
    Constructs a gaussian distribution for masses. It is enforced
    that mass_2 < mass_1 through the ConditionalLogNormal prior.

    Args:
        m1: mean of the log normal distribution for mass 1
        m2: mean of the log normal distribution for mass 2
        sigma: standard deviation of the log normal distribution

    Returns a ConditionalPriorDict
    """

    prior_dict = ConditionalPriorDict()
    prior_dict["mass_1"] = Gaussian(name="mass_1", mu=m1, sigma=sigma)

    def condition_func(reference_params, mass_1):

        return dict(
            maximum=mass_1,
            minimum=reference_params["minimum"],
            mu=reference_params["mu"],
            sigma=reference_params["sigma"],
        )

    prior_dict["mass_2"] = ConditionalTruncatedGaussian(
        name="mass_2",
        condition_func=condition_func,
        minimum=5,
        maximum=100,
        mu=m2,
        sigma=sigma,
    )

    prior_dict["luminosity_distance"] = UniformSourceFrame(
        name="luminosity_distance", minimum=100, maximum=3000, unit="Mpc"
    )
    prior_dict["dec"] = Cosine(name="dec")
    prior_dict["ra"] = Uniform(
        name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
    )

    return prior_dict
