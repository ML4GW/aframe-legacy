from pathlib import Path
from typing import List

import numpy as np
from datagen.scripts.background import deploy as deploy_background
from datagen.scripts.timeslide_waveforms import deploy as deploy_timeslides
from lal import gpstime
from typeo import scriptify

ONE_WEEK = 60 * 60 * 24 * 7


def make_outdir(outdir: Path, start: float, stop: float):
    start = gpstime.gps_to_utc(start).strftime("%m-%d-%Y")
    stop = gpstime.gps_to_utc(stop).strftime("%m-%d-%Y")
    outdir = outdir / f"{start}_{stop}"
    outdir.mkdir(exist_ok=True, parents=True)

    (outdir / "log").mkdir(exist_ok=True, parents=True)
    return outdir


@scriptify
def main(
    # general args for background and timeslides
    test_stop: float,
    intervals: List[int],
    channel: str,
    sample_rate: float,
    ifos: List[str],
    state_flag: str,
    min_segment_length: float,
    max_segment_length: float,
    duration: float,
    datadir: Path,
    # timeslide waveform specific args
    Tb: float,
    shifts: List[int],
    spacing: float,
    buffer: float,
    prior: str,
    minimum_frequency: float,
    reference_frequency: float,
    waveform_duration: float,
    waveform_approximant: str,
    highpass: float,
    snr_threshold: float,
    psd_length: float,
    # condor args
    accounting_group: str,
    accounting_group_user: str,
    request_memory: int = 32768,
    request_disk: int = 1024,
    verbose: bool = False,
):

    # TODO: ensure not in between O3a and O3b
    intervals = np.array(intervals)
    intervals *= ONE_WEEK

    for cadence in intervals:
        start, stop = test_stop + cadence, test_stop + cadence + duration
        out = make_outdir(datadir, start, stop)
        deploy_background(
            start - ONE_WEEK / 7,
            start,
            stop,
            min_segment_length,
            min_segment_length,
            ifos,
            sample_rate,
            channel,
            state_flag,
            out,
            out / "log",
            accounting_group,
            accounting_group_user,
            max_segment_length,
            request_memory,
            request_disk,
        )

        deploy_timeslides(
            start,
            stop,
            state_flag,
            Tb,
            ifos,
            shifts,
            spacing,
            buffer,
            min_segment_length,
            prior,
            minimum_frequency,
            reference_frequency,
            sample_rate,
            waveform_duration,
            waveform_approximant,
            highpass,
            snr_threshold,
            psd_length,
            out,
            out,
            out / "log",
            accounting_group_user,
            accounting_group,
            request_memory=6000,
            request_disk=1024,
        )
