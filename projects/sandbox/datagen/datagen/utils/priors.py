from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import h5py
import numpy as np
from bilby.core.prior import (
    ConditionalPowerLaw,
    ConditionalPriorDict,
    Constraint,
    Cosine,
    Interped,
    PowerLaw,
    Sine,
    Uniform,
)
from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame


# generic parent class for implementing priors
# Use frozen dataclass since this will make it
# pickleable and therefore usable with multiprocessing
@dataclass(frozen=True)
class Prior:
    # Conditional to allow for possibility of conditional priors
    prior: ConditionalPriorDict
    parameter_file: Path = None

    @staticmethod
    def build() -> ConditionalPriorDict:
        raise NotImplementedError

    @classmethod
    def create(cls):
        prior = cls.build()
        return cls(prior)

    def read_priors_from_file(self):
        with h5py.File(self.parameter_file, "r") as f:
            for key in list(f.keys()):
                param_grid = f[key]["param_grid"][:]
                p_vals = f[key]["p_vals"][:]
                self.prior[key] = Interped(param_grid, p_vals, name=key)

    def sample(self, N: int) -> Dict[str, Sequence[float]]:
        return self.prior.sample(N)


class UniformExtrinsicParametersPrior(Prior):
    @staticmethod
    def build():
        prior = ConditionalPriorDict()
        prior["dec"] = Cosine(name="dec")
        prior["ra"] = Uniform(
            name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
        )
        prior["theta_jn"] = 0
        prior["phase"] = 0

        return prior


class NonSpinBBHPrior(UniformExtrinsicParametersPrior):
    @staticmethod
    def build():
        prior = UniformExtrinsicParametersPrior.build()
        prior["mass_1"] = Uniform(
            name="mass_1", minimum=5, maximum=100, unit=r"$M_{\odot}$"
        )
        prior["mass_2"] = Uniform(
            name="mass_2", minimum=5, maximum=100, unit=r"$M_{\odot}$"
        )
        prior["mass_ratio"] = Constraint(
            name="mass_ratio", minimum=0.2, maximum=5.0
        )
        prior["luminosity_distance"] = UniformSourceFrame(
            name="luminosity_distance", minimum=100, maximum=3000, unit="Mpc"
        )
        prior["psi"] = 0
        prior["a_1"] = 0
        prior["a_2"] = 0
        prior["tilt_1"] = 0
        prior["tilt_2"] = 0
        prior["phi_12"] = 0
        prior["phi_jl"] = 0

        return prior


class EndO3RatesAndPopsPrior(UniformExtrinsicParametersPrior):
    @staticmethod
    def condition_func(reference_params, mass_1):
        return dict(
            alpha=reference_params["alpha"],
            minimum=reference_params["minimum"],
            maximum=mass_1,
        )

    @staticmethod
    def build():
        prior = UniformExtrinsicParametersPrior.build()
        prior["mass_1"] = PowerLaw(
            name="mass_1",
            alpha=-2.35,
            minimum=2,
            maximum=100,
            unit=r"$M_{\odot}",
        )
        prior["mass_2"] = ConditionalPowerLaw(
            name="mass_2",
            condition_func=EndO3RatesAndPopsPrior.condition_func,
            alpha=1,
            minimum=2,
            maximum=100,
            unit=r"$M_{\odot}",
        )
        prior["luminosity_distance"] = UniformComovingVolume(
            name="luminosity_distance", minimum=100, maximum=15000, unit="Mpc"
        )
        prior["psi"] = 0
        prior["a_1"] = Uniform(name="a_1", minimum=0, maximum=0.998)
        prior["a_2"] = Uniform(name="a_2", minimum=0, maximum=0.998)
        prior["tilt_1"] = Sine(name="tilt_1", unit="rad")
        prior["tilt_2"] = Sine(name="tilt_2", unit="rad")
        prior["phi_12"] = Uniform(
            name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic"
        )
        prior["phi_jl"] = 0

        return prior


extrinsic_params = UniformExtrinsicParametersPrior.create()
nonspin_bbh = NonSpinBBHPrior.create()
end_o3_ratesandpops = EndO3RatesAndPopsPrior.create()
power_law_break_dip = extrinsic_params.read_priors_from_file("param_file.h5")
