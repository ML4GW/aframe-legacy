import os
from collections.abc import Iterable

import h5py
import numpy as np
from gwpy.segments import Segment, SegmentList
from gwpy.timeseries import TimeSeries
from hermes.typeo import typeo

from bbhnet.injection import inject_signals


@typeo
def main(
    start: int,
    stop: int,
    outdir: str,
    prior_file: str,
    n_samples: int,
    n_slides: int = 600,
    shift: float = 0.5,
    ifos: Iterable[str] = ["H1", "L1"],
    seg_length: int = 1024,
    fmin: float = 20,
    waveform_duration: float = 8,
    snr_range: Iterable[float] = [25, 50],
):
    """Generates timeslides of background and background+injections.
    Also saves the original and injected timeseries as frame files.

    Args:
        start: starting GPS time of time period
        stop: ending GPS time of time period
        outdir: base directory where other directories will be created
        prior_file: a .prior file containing the priors for the GW simulation
        n_samples: number of signals to simulate per file
        n_slides: number of timeslides
        shift: time in seconds of each slide
        ifos: pair of interferometers to be compared
        seg_length: length in seconds of each separate file
        fmin: min frequency for highpass filter, used for simulating
        waveform_duration: length of injected waveforms
        snr_range: desired signal SNR range

    """

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for t0 in np.arange(start, stop, seg_length):

        tf = min(t0 + seg_length, stop)

        # Go to the next seg_length segment if segment contains a known GW
        gw_times = np.loadtxt("O3b_GW_times.txt")
        gw_seg_list = SegmentList([Segment(t, t) for t in gw_times])
        if gw_seg_list.intersects_segment(Segment(t0, tf)):
            continue

        # Go to the next seg_length segment if either ifo has nan values
        has_nans = False
        background = {}
        for ifo in ifos:
            background[ifo] = TimeSeries.fetch_open_data(ifo, t0, tf)
            background[ifo].name = ifo
            if np.isnan(np.sum(background[ifo].value)):
                has_nans = True
                break

        if has_nans:
            continue

        # Write data to gwf so signals can be added
        orig_path = os.path.join(outdir, "original")
        if not os.path.exists(orig_path):
            os.mkdir(orig_path)

        fpaths = []
        file_dur = tf - t0
        for ifo in ifos:
            fname = f"{ifo}_{t0}_{file_dur}.gwf"
            background[ifo].write(os.path.join(orig_path, fname))
            fpaths.append(os.path.join(orig_path, fname))

        inj_path = os.path.join(outdir, "injected")
        if not os.path.exists(inj_path):
            os.mkdir(inj_path)

        # Adjust n_samples for last segment to match density of others
        if tf == stop:
            n_samples = int(n_samples * file_dur / seg_length)

        outpaths, _ = inject_signals(
            frame_files=fpaths,
            channels=ifos,
            ifos=ifos,
            prior_file=prior_file,
            n_samples=n_samples,
            outdir=inj_path,
            fmin=fmin,
            waveform_duration=waveform_duration,
            snr_range=snr_range,
        )

        # Grab this value before converting TimeSeries to just values
        sample_rate = background[ifos[0]].sample_rate.value

        injected = {}
        for i, ifo in enumerate(ifos):
            background[ifo] = background[ifo].value
            injected[ifo] = TimeSeries.read(outpaths[i], ifo).value

        for ts in np.linspace(0, shift * (n_slides - 1), num=n_slides):

            # Create the desired structure in the output directory
            ts_path = os.path.join(outdir, "dt-{:.1f}".format(ts))
            orig_path = os.path.join(ts_path, "original")
            inj_path = os.path.join(ts_path, "injected")

            if not os.path.exists(ts_path):
                os.mkdir(ts_path)

            if not os.path.exists(orig_path):
                os.mkdir(orig_path)

            if not os.path.exists(inj_path):
                os.mkdir(inj_path)

            # Write strain timeseries into appropriate locations
            orig_file_path = os.path.join(orig_path, f"{t0}_{file_dur}.hdf5")
            inj_file_path = os.path.join(inj_path, f"{t0}_{file_dur}.hdf5")
            with h5py.File(orig_file_path, "w") as f:
                for ifo in ifos:
                    f.create_dataset(ifo, data=background[ifo])

            with h5py.File(inj_file_path, "w") as f:
                for ifo in ifos:
                    f.create_dataset(ifo, data=injected[ifo])

            # Roll the strain data of the second ifo by the desired shift
            background[ifos[1]] = np.roll(
                background[ifos[1]], int(shift * sample_rate)
            )
            injected[ifos[1]] = np.roll(
                injected[ifos[1]], int(shift * sample_rate)
            )


if __name__ == "__main__":
    main()
