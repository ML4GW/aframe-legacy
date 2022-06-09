import numpy as np
import torch
from gwpy.signal.filter_design import fir_from_transfer
from gwpy.timeseries import TimeSeries

from bbhnet.data.transforms.transform import Transform

DEFAULT_FFTLENGTH = 2


class WhiteningTransform(Transform):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        fftlength: float = DEFAULT_FFTLENGTH,
    ) -> None:
        super().__init__()
        self.num_ifos = num_ifos
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.fftlength = fftlength

        # initialize the parameter with 0s, then fill it out later
        self.time_domain_filter = self.add_parameter(
            np.zeros(num_ifos, int(fftlength * sample_rate))
        )
        self.window = torch.hann_window(int(fftlength * sample_rate))

    def to(self, device: torch.device):
        """
        Quick override of device placement to ensure
        that our window, which is a _tensor_ and not
        a _parameter_, gets moved to the proper device
        """
        super().to(device)
        self.window.to(self.time_domain_filter.device)

    def fit(self, X: torch.Tensor) -> None:
        """
        Build a whitening time domain filter from a set
        of ASDs. TODO: should this be a single tensor
        stacking all the backgrounds for consistency?
        """
        if X.ndim != 2:
            raise ValueError(
                "Expected background used to fit WhiteningTransform "
                "to have 2 dimensions, but found {}".format(X.ndim)
            )
        if len(X) != self.time_domain_filter.shape[0]:
            raise ValueError(
                "Expected to fit whitening transform on {} backgrounds, "
                "but was passed {}".format(
                    self.time_domain_filter.shape[0], len(X)
                )
            )

        ntaps = int(self.fftlength * self.sample_rate)
        tdfs = []
        for x in X.cpu().numpy():
            ts = TimeSeries(x, dt=1 / self.sample_rate)
            asd = ts.asd(
                fftlength=self.fftlength, window="hanning", method="median"
            )
            asd = asd.interpolate(1 / self.kernel_length).value
            tdf = fir_from_transfer(
                1 / asd, ntaps=ntaps, window="hanning", ncorner=0
            )
            tdfs.append(tdf)

        tdf = np.stack(tdfs)
        self.set_value(self.time_domain_filter, tdf)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # do a constant detrend along the time axis,
        # transposing to ensure that the last two dimensions
        # of the original and dimension-reduced tensors match.
        # TODO: will using X.mean(axis=-1, keepdims=True)
        # allow us to avoid these transposes?
        X = X.transpose(2, 0)
        X = X - X.mean(axis=0)
        X = X.transpose(0, 2)
        X *= self.window

        # convolve the detrended data with the time-domain
        # filters constructed during initialization from
        # the background data, using groups to ensure that
        # the convolution is performed independently for
        # each interferometer channel
        X = torch.nn.functional.conv1d(
            X, self.time_domain_filter, groups=self.num_ifos, padding="same"
        )

        # scale by sqrt(2 / sample_rate) for some inscrutable
        # signal processing reason beyond my understanding
        return X * (2 / self.sample_rate) ** 0.5
