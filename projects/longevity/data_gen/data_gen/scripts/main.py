import inspect
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from datagen.scripts.background import deploy as deploy_background
from datagen.scripts.glitches import main as generate_glitches
from datagen.scripts.timeslide_waveforms import deploy as deploy_timeslides
from datagen.scripts.waveforms import main as generate_waveforms
from lal import gpstime
from typeo import scriptify

from aframe.logging import configure_logging

ONE_WEEK = 60 * 60 * 24 * 7


def make_outdir(outdir: Path, start: float, stop: float):
    start = gpstime.gps_to_utc(start).strftime("%m-%d-%Y")
    stop = gpstime.gps_to_utc(stop).strftime("%m-%d-%Y")
    outdir = outdir / f"{start}_{stop}"
    outdir.mkdir(exist_ok=True, parents=True)

    (outdir / "log").mkdir(exist_ok=True, parents=True)
    return outdir


def deploy_background_wrapper(
    train_start: float,
    train_stop: float,
    test_stop: float,
    minimum_train_length: float,
    minimum_test_length: float,
    ifos: List[str],
    sample_rate: float,
    channel: str,
    state_flag: str,
    datadir: Path,
    logdir: Path,
    accounting_group: str,
    accounting_group_user: str,
    max_segment_length: float = 20000,
    request_memory: int = 32768,
    request_disk: int = 1024,
    force_generation: bool = False,
    verbose: bool = False,
):
    deploy_background(
        train_start=train_start,
        train_stop=train_stop,
        test_stop=test_stop,
        minimum_train_length=minimum_train_length,
        minimum_test_length=minimum_test_length,
        ifos=ifos,
        sample_rate=sample_rate,
        channel=channel,
        state_flag=state_flag,
        datadir=datadir,
        logdir=logdir,
        accounting_group=accounting_group,
        accounting_group_user=accounting_group_user,
        max_segment_length=max_segment_length,
        request_memory=request_memory,
        request_disk=request_disk,
        force_generation=force_generation,
        verbose=verbose,
    )
    return train_stop, test_stop, datadir


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
    basedir: Path,
    # timeslide waveform specific args
    Tb: float,
    shifts: List[int],
    spacing: float,
    buffer: float,
    prior: Callable,
    minimum_frequency: float,
    reference_frequency: float,
    waveform_duration: float,
    waveform_approximant: str,
    highpass: float,
    snr_threshold: float,
    psd_length: float,
    # waveform generation args
    num_signals: int,
    # condor / misc args
    accounting_group: str,
    accounting_group_user: str,
    request_memory: int = 32768,
    request_disk: int = 1024,
    verbose: bool = False,
    force_generation: bool = False,
    seed: Optional[int] = None,
):

    basedir.mkdir(exist_ok=True, parents=True)
    configure_logging(basedir / "datagen.log", verbose=verbose)
    # TODO: ensure not in between O3a and O3b
    intervals = np.array(intervals)
    intervals *= ONE_WEEK

    pool = ProcessPoolExecutor(8)
    prior_str = inspect.getmodule(prior).__name__ + "." + prior.__name__
    # launch background generation jobs that will query both:
    # training data: (for training new models) and
    # testing data: for testing both original and retrained models
    background_futures = []
    with pool as executor:
        for cadence in intervals:

            # get train and test intervals from cadence
            start, stop = test_stop + cadence, test_stop + cadence + duration
            train_start = start - ONE_WEEK
            train_stop = start

            out = make_outdir(basedir, start, stop)
            datadir = out / "data"

            logging.info(f"Deploying background generation for {out}")
            args = [
                train_start,  # re-train using one week
                train_stop,
                stop,
                min_segment_length,
                min_segment_length,
                ifos,
                sample_rate,
                channel,
                state_flag,
                datadir,
                datadir / "log",
                accounting_group,
                accounting_group_user,
                max_segment_length,
                request_memory,
                request_disk,
            ]
            future = executor.submit(deploy_background_wrapper, *args)
            background_futures.append(future)

            # deploy glitch generation jobs that will
            # be used for investigating glitch rates of intervals
            logging.info(f"Deploying glitch generation for {out}")
            args = [
                5,  # snr thresh
                train_start,
                train_stop,
                stop,
                3.3166,  # qmin
                108,  # qmax
                highpass,
                0.5,
                124,  # segment duration
                64,  # chunk duration
                4,  # overlap
                0.2,  # mismatch max
                2,  # window
                datadir,
                datadir / "log",
                "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",  # same as open data
                "HOFT_CLEAN_SUB60HZ_C01",  # frame type for above channel
                sample_rate,
                # omicron doesn't support open data
                "DCS-ANALYSIS_READY_C01:1",
                ifos,
                4096,
                True,
                False,
                True,
            ]
            future = executor.submit(generate_glitches, *args)

        # next, launch timeslide waveform generation jobs
        # that will be used to test original and retrained models
        futures = []
        for future in as_completed(background_futures):
            start, stop, datadir = future.result()
            logging.info(f"Deploying timeslides waveform generation for {out}")
            args = [
                start,
                stop,
                Tb,
                ifos,
                shifts,
                spacing,
                buffer,
                min_segment_length,
                prior_str,
                minimum_frequency,
                reference_frequency,
                sample_rate,
                waveform_duration,
                waveform_approximant,
                highpass,
                snr_threshold,
                psd_length,
                datadir,
                datadir,
                datadir / "log",
                accounting_group_user,
                accounting_group,
                6000,
                1024,
                force_generation,
                verbose,
                seed,
            ]
            future = executor.submit(deploy_timeslides, *args)
            futures.append(future)

        for cadence in intervals:
            start, stop = test_stop + cadence, test_stop + cadence + duration
            interval = make_outdir(basedir, start, stop)

            datadir = interval / "data"
            logging.info(f"Deploying waveform generation for {interval}")
            args = [
                prior,
                num_signals,
                datadir / "train",
                datadir / "log",
                reference_frequency,
                minimum_frequency,
                sample_rate,
                waveform_duration,
                waveform_approximant,
                force_generation,
                verbose,
                seed,
            ]
            future = executor.submit(generate_waveforms, *args)
            futures.append(future)

    # wait for all the jobs to finish
    for future in as_completed(futures):
        if future.exception() is not None:
            logging.info(future.exception())
        logging.info("future done")
        continue
