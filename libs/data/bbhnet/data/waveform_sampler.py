from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
from bilby.core.prior import Cosine, PriorDict, Uniform
from gwpy.frequencyseries import FrequencySeries

from bbhnet.data.utils import sample_kernels
from bbhnet.injection import project_raw_gw

PRIORS = PriorDict(
    {
        "ra": Uniform(minimum=0, maximum=2 * np.pi),
        "dec": Cosine(),
        "psi": Uniform(minimum=0, maximum=np.pi),
    }
)


@dataclass
class _DummyWaveformGenerator:
    sampling_frequency: float
    duration: float


class WaveformSampler:
    def __init__(
        self,
        dataset: str,
        sample_rate: float,
        min_snr: float,
        max_snr: float,
        highpass: Optional[float] = 20,
    ):
        with h5py.File(dataset, "r") as f:
            self.waveforms = f["signals"][:]

        self.priors = PRIORS.copy()

        self.df = 1 / (sample_rate * self.waveforms.shape[1])
        self.sample_rate = sample_rate
        self.min_snr = min_snr
        self.max_snr = max_snr

        freqs = np.arange(0, sample_rate // 2 + self.df, self.df)
        if highpass is not None:
            self.mask = freqs <= highpass
        else:
            # lazy way of just making everything true
            self.mask = freqs >= 0

        # initialize some attributes that need
        #  to be fit to a particular background
        self.background_asds = self.ifos = None

    def fit(
        self, t0: float, tf: float, *background_asds: FrequencySeries
    ) -> None:
        ifos = [asd.channel.split(":")[0] for asd in background_asds]
        background_asds = [asd.interpolate(self.df) for asd in background_asds]
        background_asds = np.stack([asd.value for asd in background_asds])

        # TODO: replace with 1s here or clip to a nonzero value?
        background_asds = np.clip(background_asds, 1e-6, np.inf)

        self.priors["geocent_time"] = Uniform(minimum=t0, maximum=tf)
        self.background_asd = background_asds
        self.ifos = ifos

    def compute_snrs(self, signals: np.ndarray) -> np.ndarray:
        ffts = np.fft.rfft(signals, axis=-1)
        snrs = 2 * np.abs(ffts) / self.background_asd
        snrs = self.df * snrs**2
        return snrs[self.mask].sum(axis=-1) ** 0.5

    def reweight_snrs(self, signals: np.ndarray) -> np.ndarray:
        snrs = self.compute_snrs(signals)
        snrs = (snrs**2).sum(axis=1) ** 0.5

        target_snrs = np.random.uniform(
            self.min_snr, self.max_snr, size=len(snrs)
        )
        weights = target_snrs / snrs
        snrs = snrs.transpose(1, 2, 0) * weights
        return snrs.transpose(2, 0, 1)

    def sample(self, N: int, size: int) -> np.ndarray:
        if self.background_asd is None:
            raise ValueError(
                "Must fit WaveformGenerator to background asd before sampling"
            )

        # sample some waveform indices to inject as well
        # as sky localization parameters for computing
        # the antenna response in real-time
        idx = np.random.choice(len(self.waveforms), size=N, replace=False)
        sample_params = self.priors.sample(N)

        # initialize the output array and a dummy object
        # which has a couple attributes expected by the
        # argument passed to project_raw_gw
        # TODO: project_raw_gw should accept these arguments on their own
        signals = np.zeros((N, len(self.ifos), size))
        waveform_generator = _DummyWaveformGenerator(
            self.sample_rate, size // self.sample_rate
        )

        # for each one of the interferometers used in
        # the background asds passed to `.fit`, compute
        # its response to the waveform given the sky
        # localization parameters
        for i, ifo in enumerate(self.ifos):
            signal = project_raw_gw(
                self.waveforms[idx],
                sample_params,
                waveform_generator,
                ifo,
                get_snr=False,
            )
            signals[:, i] = signal

        # scale the amplitudes of the signals so that
        # their RMS SNR falls in an acceptable range
        signals = self.reweight_snrs(signals)

        # randomly sample kernels from these signals
        return sample_kernels(signals, size=size)
