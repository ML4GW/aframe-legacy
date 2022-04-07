from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
import torch
from gwpy.signal.filter_design import fir_from_transfer
from gwpy.timeseries import TimeSeries

from bbhnet.injection import project_raw_gw

DEFAULT_FFTLENGTH = 2


@dataclass
class DummyWaveformGenerator:
    sampling_frequency: float
    duration: float


def _build_time_domain_filter(
    timeseries: np.ndarray, sample_rate: float, kernel_length: float
) -> np.ndarray:
    """
    Create a time-domain filter for whitening using the
    indicated timeseries as background. Replicating the
    behavior of `gwpy.timeseries.TimeSeries.whiten`.
    Args:
        timeseries:
            Background timeseries whose spectrum to
            use for whitening
        sample_rate:
            Data rate of timeseries
        kernel_length:
            Length of data that will eventually be used
            to be whitened by this filter
    Returns:
        Time domain filter to convolve with sampled
        training data
    """

    ts = TimeSeries(timeseries, dt=1 / sample_rate)
    asd = ts.asd(
        fftlength=DEFAULT_FFTLENGTH, window="hanning", method="median"
    )

    asd = asd.interpolate(1 / kernel_length)
    ntaps = int(DEFAULT_FFTLENGTH * sample_rate)
    return fir_from_transfer(
        1 / asd.value, ntaps=ntaps, window="hanning", ncorner=0
    )


class RandomWaveformDataset:
    def __init__(
        self,
        hanford_background: str,
        livingston_background: str,
        kernel_length: float,
        sample_rate: float,
        batch_size: int,
        batches_per_epoch: int,
        waveform_dataset: Optional[str] = None,
        waveform_frac: float = 0,
        glitch_dataset: Optional[str] = None,
        glitch_frac: float = 0,
        device: torch.device = "cuda",
    ) -> None:
        """Iterable dataset which can sample and inject auxiliary data
        Iterable dataset for use with torch.data.DataLoader which
        generates tensors of background data from the two LIGO
        interferometers. Optionally can inject simulated waveforms
        and insert real glitch data which are sampled from HDF5
        datasets.
        Background data is sample uniformly and independently for
        both interferometers, simulating arbitrary time-shifts.
        The cost of this is that we abandon the traditional notion
        of an "epoch" as "one full pass through the dataset", since
        the sampling makes no attempt to exclude kernels which may
        have been sampled recently. As such, the `batches_per_epoch`
        kwarg is used to determine how many batches to produce
        before to raising a `StopIteration` to move on to tasks
        like validation.
        Args:
            hanford_background:
                Path to HDF5 file containing background data for
                the Hanford interferometer under the dataset
                key `"hoft"`. Assumed to be sampled at rate
                `sample_rate`.
            livingston_background:
                Path to HDF5 file containing background data for
                the Livingston interferometer under the datset
                key `"hoft"`. Assumed to be sampled at rate
                `sample_rate`. Must contain the same amount of
                data as `hanford_background`.
            kernel_length:
                The length, in seconds, of each batch element
                to produce during iteration.
            sample_rate:
                The rate at which all relevant input data has
                been sampled
            batch_size:
                Number of samples to produce during at each
                iteration
            batches_per_epoch:
                The number of batches to produce before raising
                a `StopIteration` while iterating
            waveform_dataset:
                Path to HDF5 file containing pure simulated
                waveforms to inject into the background data.
                Still not sure what this will need to look like.
                The trigger for each waveform should sit in the
                middle of the time axis of the array containing
                the waveforms.
            waveform_frac:
                The fraction of each batch that should consist
                of injected waveforms, and be marked with a
                `1.` in the target tensor produced during iteration.
            glitch_dataset:
                Path to HDF5 file containing glitches from each
                interferometer contained in `"hanford"` and
                `"livingston"` dataset keys. Data should be
                sampled at rate `sample_rate`. Glitch triggers
                should sit in the middle of the time axis of
                the array containing the glitch timeseries.
                Glitches will be randomly sampled and inserted in
                place of the corresponding interferometer channel
                background at data-loading time.
            glitch_frac:
                The fraction of each batch that should consist
                of inserted glitches, marked as `0.` in the
                target tensor produced during iteration
            device:
                The device on which to host all the relevant
                torch tensors.
        """

        # sanity check our fractions
        assert 0 <= waveform_frac <= 1
        assert 0 <= glitch_frac <= 1

        self.sample_rate = sample_rate
        self.kernel_size = int(kernel_length * sample_rate)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.device = device

        # load in the background data and build time-domain
        # filters for whitening using them
        # TODO: maybe these are gwf and we resample?
        with h5py.File(hanford_background, "r") as f:
            hanford_bkgrd = f["hoft"][:]
            hanford_filter = _build_time_domain_filter(
                hanford_bkgrd, sample_rate, kernel_length
            )

            # move everything onto the GPU up front so that
            # we don't have to pay for transfer time later.
            # If our datasets are on the scale of ~GBs this
            # shouldn't be a problem, esp. for the current
            # size of BBHNet
            self.hanford_background = torch.Tensor(hanford_bkgrd).to(device)
        with h5py.File(livingston_background, "r") as f:
            livingston_bkgrd = f["hoft"][:]

            # ensure that we have the same amount of
            # data from both detectors
            assert len(livingston_bkgrd) == len(hanford_bkgrd)

            livingston_filter = _build_time_domain_filter(
                livingston_bkgrd, sample_rate, kernel_length
            )
            self.livingston_background = torch.Tensor(livingston_bkgrd).to(
                device
            )

        # move the filters to a single torch Tensor
        # on the desired device, adding a dummy dimension
        # in the middle for compatability with Torch's
        # conv op's expectations
        whitening_filter = np.stack([hanford_filter, livingston_filter])
        self.whitening_filter = torch.Tensor(whitening_filter[:, None]).to(
            device
        )

        # create a window for applying the time domain filter
        # at data loading time. Since the filter will have the
        # same size as the timeseries at data loading time,
        # we can use the window as-is without having to appply
        # it just to the edges of the timeseries
        self.whitening_window = torch.hann_window(
            whitening_filter.shape[-1], device=device
        )

        # load in any waveforms if we specified them
        # TODO: what will the actual field name be?
        # TODO: will these need to be resampled?
        if waveform_dataset is not None:
            # if we specified waveforms, make sure we're
            # actually planning on using them
            assert waveform_frac > 0
            self.num_waveforms = max(1, int(waveform_frac * batch_size))

            with h5py.File(waveform_dataset, "r") as f:
                # should have shape:
                # (num_waveforms, 2, sample_rate * num_seconds)
                # where 2 is for each detector and num_seconds
                # is however long we had bilby make the waveforms
                self.waveforms = f["signals"][:]
            self.waveform_generator = DummyWaveformGenerator(
                sample_rate, kernel_length
            )
        else:
            # likewise, ensure that we didn't indicate that
            # we expected any waveforms in the batch
            assert waveform_frac == 0
            self.num_waveforms = 0
            self.waveforms = None

        # load in any glitches if we specified them
        # TODO: what will the actual field name be?
        # TODO: will these need to be resampled?
        if glitch_dataset is not None:
            # if we specified glitches, make sure we're
            # actually planning on using them
            assert glitch_frac > 0
            self.num_glitches = max(1, int(glitch_frac * batch_size))

            with h5py.File(glitch_dataset, "r") as f:
                self.hanford_glitches = torch.Tensor(f["hanford"][:]).to(
                    device
                )
                self.livingston_glitches = torch.Tensor(f["livingston"][:]).to(
                    device
                )
        else:
            # likewise, ensure that we didn't indicate that
            # we expected any glitches in the batch
            assert glitch_frac == 0
            self.num_glitches = 0
            self.hanford_glitches = self.livingston_glitches = None

        # make sure that we have at least _some_
        # pure background in each batch
        assert (self.num_waveforms + self.num_glitches) < batch_size

    def sample_from_array(self, array: torch.Tensor, N: int) -> torch.Tensor:
        """Sample kernels from an array of timeseries
        For an array of timeseries of shape
        (num_timeseries, self.sample_rate * length),
        where length is some characteristic length of the
        type of event contained in each timeseries, and the
        event in question is assumed to "trigger" in the
        middle of each timeseries.
        First uniformly samples `N` timeseries to sample
        kernels from, then uniformly samples a kernel
        from that timeseries that contains the trigger.
        """

        # sample `N` timeseries to sample kernels from
        idx = np.random.choice(len(array), size=N, replace=False)

        # for each timeseries, grab a random kernel-sized
        # TODO: is there a good way to do this with array ops?
        sample_start = min(
            array.shape[-1] // 2 - 1, array.shape[-1] - self.kernel_size
        )

        samples = []
        for i in idx:
            # sample from within a kernel's length of
            # the center of array, where it is assumed
            # that the "trigger" of the relevant event will live
            start = np.random.randint(sample_start)
            stop = start + self.kernel_size

            # unfortunately can't think of a cleaner
            # way to make sure wer'e slicing from the
            # last dimension
            if len(array.shape) == 2:
                samples.append(array[i, start:stop])
            else:
                samples.append(array[i, :, start:stop])

        # stack the samples into a tensor
        return torch.stack(samples)

    def sample_from_background(self, independent: bool = True):
        """Sample a batch of kernels from the background data
        Randomly sample kernels from the interferometer
        background timeseries in a uniform manner. If
        `independent == True`, kernels will be sampled
        from each interferometer independently and then
        concatenate to simulate fully random time-shifting.
        """

        # sample kernel start indices uniformly
        # from the hanford background data ensuring
        # that the corresponding stop index will fall
        # within the timeseries
        hanford_start = np.random.choice(
            len(self.hanford_background) - self.kernel_size,
            size=self.batch_size,
            replace=False,
        )

        # if sampling indepdently, create a second set
        # of start indices for the livingston background
        # data. Otherwise, use the same indices as for hanford
        if independent:
            livingston_start = np.random.choice(
                len(self.livingston_background) - self.kernel_size,
                size=self.batch_size,
                replace=False,
            )
        else:
            livingston_start = hanford_start

        # grab kernels for each start index from the background
        # TODO: is there a good way to do this with array ops?
        X = []
        for h_idx, l_idx in zip(hanford_start, livingston_start):
            hanford = self.hanford_background[h_idx : h_idx + self.kernel_size]
            livingston = self.livingston_background[
                l_idx : l_idx + self.kernel_size
            ]
            X.extend([hanford, livingston])

        # stack everything to make a (batch_size * 2, sample_rate)
        # sized tensor, then reshape to move the two
        # interferometers to the 1st dimension
        X = torch.stack(X, dim=0).reshape(self.batch_size, 2, -1)
        return X

    def _sample_ra(self, num_samples):
        raise NotImplementedError

    def _sample_dec(self, num_samples):
        raise NotImplementedError

    def _sample_psi(self, num_samples):
        raise NotImplementedError

    def _sample_geocent_time(self, num_samples):
        raise NotImplementedError

    def inject_waveforms(self, background, waveforms):
        sample_params = {
            "ra": self._sample_ra(len(background)),
            "dec": self._sample_dec(len(background)),
            "psi": self._sample_psi(len(background)),
            "geocent_time": self._sample_geocent_time(len(background)),
        }

        signals = np.zeros((len(background), 2, self.kernel_size))
        for i, ifo in enumerate(["h1", "l1"]):
            signal = project_raw_gw(
                waveforms,
                sample_params,
                self.waveform_generator,
                ifo,
                get_snr=False,
            )
            signals[:, i] = signal
        return background + torch.Tensor(signals).to(self.device)

    def whiten(self, X: torch.Tensor) -> torch.Tensor:
        """Detrend and time-domain filter an array
        Use the time-domain filters built from the
        background data used to initialize the dataset
        to whiten an array of data.
        """

        # do a constant detrend along the time axis,
        # transposing to ensure that the last two dimensions
        # of the original and dimension-reduced tensors match.
        # TODO: will using X.mean(axis=-1, keepdims=True)
        # allow us to avoid these transposes?
        X = X.transpose(2, 0)
        X = X - X.mean(axis=0)
        X = X.transpose(0, 2)
        X *= self.whitening_window

        # convolve the detrended data with the time-domain
        # filters constructed during initialization from
        # the background data, using groups to ensure that
        # the convolution is performed independently for
        # each interferometer channel
        X = torch.nn.functional.conv1d(
            X, self.whitening_filter, groups=2, padding="same"
        )

        # scale by sqrt(2 / sample_rate) for some inscrutable
        # signal processing reason beyond my understanding
        return X * (2 / self.sample_rate) ** 0.5

    def __iter__(self):
        self._batch_idx = 0
        return self

    def __next__(self):
        if self._batch_idx >= self.batches_per_epoch:
            raise StopIteration

        # create an array of all background
        X = self.sample_from_background(independent=True)

        # create a target tensor, marking all
        # the glitch data as 0.
        y = torch.zeros((self.batch_size,))

        # replace some of this data with glitches if
        # we have glitch data to use
        if self.hanford_glitches is not None:
            # break up the number of glitches randomly
            # between hanford and livingston
            num_hanford = np.random.randint(self.num_glitches)
            num_livingston = self.num_glitches - num_hanford

            # replace the hanford channel of the
            # existing background data with some
            # sampled hanford glitches
            if num_hanford > 0:
                hanford_glitches = self.sample_from_array(
                    self.hanford_glitches, num_hanford
                )
                X[:num_hanford, 0] = hanford_glitches

            # replace the livingston channel of the existing
            # background data with some sampled livingston
            # glitches
            if num_livingston > 0:
                livingston_glitches = self.sample_from_array(
                    self.livingston_glitches, num_livingston
                )
                X[num_hanford : self.num_glitches, 1] = livingston_glitches

        # inject waveforms into the background if we have
        # generated waveforms to sample from
        if self.waveforms is not None:
            waveforms = self.sample_from_array(
                self.waveforms, self.num_waveforms
            )
            self.inject_waveforms(X[-self.num_waveforms :], waveforms)
            y[-self.num_waveforms :] = 1

        X = self.whiten(X)
        self._batch_idx += 1
        return X, y
