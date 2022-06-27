from pathlib import Path
from typing import Iterable, List, Optional

import gwdatafind
import numpy as np
from gwpy.segments import (
    DataQualityDict,
    Segment,
    SegmentList,
    SegmentListDict,
)
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from hermes.typeo import typeo

from bbhnet.injection import inject_signals_into_timeslide
from bbhnet.io import h5
from bbhnet.io.timeslides import TimeSlide


def circular_shift_segments(
    start: float, stop: float, shift: float, segments: SegmentList
):
    """Takes a gwpy SegmentList object and performs a circular time shift.

    For example:
        circ_shifted_segments = circular_shift_segments(
            start=0,
            stop=100,
            shift = 50,
            segments = SegmentList(Segment([70, 90]))
        )

        circ_shifted_segments = SegmentList(Segment([20, 40]))

    """

    if shift < 0:
        raise NotImplementedError(
            "circularly shifting segments is not"
            " yet implemented for negative shifts"
        )

    # shift segments by specified amount
    shifted_segments = segments.shift(shift)

    # create output of circularly shifted segments
    circular_shifted_segments = SegmentList([])

    # create full segment from start to stop
    # to use for deciding if part of a segment
    # needs to wrap around to the front
    full_segment = Segment([start, stop])

    for segment in shifted_segments:
        seg_start, seg_stop = segment

        # if segment is entirely between
        # start and stop just append
        if segment in full_segment:
            circular_shifted_segments.append(segment)

        # the entire segment got shifted
        # past the stop, so loop the segment around
        elif seg_start > stop:

            segment = segment.shift(start - stop)
            circular_shifted_segments.append(segment)

        # only a portion of the segment got shifted to front
        # so need to split up the segment
        elif seg_stop > stop:
            first_segment = Segment([seg_start, stop])
            second_segment = Segment([start, seg_stop - stop])
            circular_shifted_segments.extend([first_segment, second_segment])

    circular_shifted_segments = circular_shifted_segments.coalesce()
    return circular_shifted_segments


@typeo
def main(
    start: int,
    stop: int,
    outdir: Path,
    prior_file: str,
    spacing: float,
    jitter: float,
    buffer: float,
    n_slides: int,
    shifts: Iterable[float],
    ifos: Iterable[str],
    file_length: int,
    fmin: float,
    sample_rate: float,
    frame_type: str,
    channel: str,
    waveform_duration: float,
    reference_frequency: float,
    waveform_approximant: str,
    snr_range: List[float],
    state_flag: Optional[str] = None,
):
    """Generates timeslides of background and background + injections.
    Also saves the original and injected timeseries as frame files.

    Args:
        start: starting GPS time of time period
        stop: ending GPS time of time period
        outdir: base directory where other directories will be created
        prior_file: a .prior file containing the priors for the GW simulation
        n_samples: number of signals to simulate per file
        n_slides: number of timeslides
        shift:
            List of shift multiples for each ifo. Will create n_slides
            worth of shifts, at multiples of shift. If 0 is passed,
            will not shift this ifo for any slide.
        ifos: pair of interferometers to be compared
        seg_length: length in seconds of each separate file
        fmin: min frequency for highpass filter, used for simulating
        waveform_duration: length of injected waveforms
        snr_range: desired signal SNR range
        gw_timesfile: path to txt file containing GPS times of GWs

    """

    outdir.mkdir(parents=True, exist_ok=True)

    # query all necessary data up front

    data = TimeSeriesDict()
    for ifo in ifos:

        files = gwdatafind.find_urls(
            site=ifo.strip("1"),
            frametype=f"{ifo}_{frame_type}",
            gpsstart=start,
            gpsend=stop,
            urltype="file",
        )
        data[ifo] = TimeSeries.read(
            files, channel=f"{ifo}:{channel}", start=start, end=stop
        )

    # if state_flag is passed,
    # query segments for each ifo.
    # a certificate is needed for this
    if state_flag:
        segments = DataQualityDict.query_dqsegdb(
            [f"{ifo}:{state_flag}" for ifo in ifos],
            start,
            stop,
        )

    else:
        # make segment from start to stop
        segments = SegmentListDict()
        for ifo in ifos:
            segments[f"{ifo}:{state_flag}"] = SegmentList(
                [Segment(start, stop)]
            )

    # create timeslides
    # for each ifo
    timeslides = [
        np.linspace(0, shift * (n_slides - 1), num=n_slides)
        for shift in shifts
    ]

    for shifts in timeslides:

        # TODO: might be overly complex naming,
        # but wanted to attempt to generalize to multi ifo
        root = outdir / f"dt-{'-'.join(shifts)}"

        # create TimeSlide object for injection
        injection_ts = TimeSlide(root=root, field="injection")

        # create TimeSlide object for raw data
        raw_ts = TimeSlide(root=root, field="raw")

        # initiate segment intersection as full
        # segment from start, stop
        intersection = SegmentList([start, stop])

        # circularly shift data
        # circularly shift segments to 'mirror' data
        for shift, ifo in zip(shifts, ifos):

            segments[f"{ifo}:{state_flag}"] = circular_shift_segments(
                start, stop, shift, segments[f"{ifo}:{state_flag}"]
            )

            shifted_data = np.roll(data[ifo].value, int(shift * sample_rate))
            data[ifo] = TimeSeries(
                shifted_data, dt=1 / sample_rate, start=start, stop=stop
            )

            # calculate intersection of circularly shifted segments
            intersection &= segments[f"{ifo}:{state_flag}"].active

        for segment in intersection:

            # crop data to segment
            segment_start, segment_stop = segment

            times = np.arange(segment_start, segment_stop, sample_rate)

            # write timeseries
            for t0 in np.arange(segment_start, segment_stop, file_length):

                tf = min(t0 + file_length, segment_stop)
                raw_datasets = {}

                for ifo in ifos:
                    raw_datasets[ifo] = data[ifo].crop(t0, tf).value

                times = np.arange(t0, tf, 1 / sample_rate)
                h5.write_timeseries(
                    raw_ts.root, prefix="raw", t=times, datasets=raw_datasets
                )

        # now inject signals into raw files;
        # this function automatically writes h5 files to TimeSlide
        # for injected data
        inject_signals_into_timeslide(
            raw_ts,
            injection_ts,
            ifos,
            prior_file,
            spacing,
            fmin,
            waveform_duration,
            sample_rate,
            fmin,
            reference_frequency,
            waveform_approximant,
            snr_range,
            buffer,
        )


if __name__ == "__main__":
    main()
