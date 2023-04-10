import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import datagen.utils.timeslide_waveforms as utils
import numpy as np
import torch
from datagen.utils.injection import generate_gw
from mldatafind.segments import query_segments
from typeo import scriptify

from bbhnet.analysis.ledger.injections import LigoResponseSet
from bbhnet.deploy import condor
from bbhnet.logging import configure_logging
from ml4gw.gw import (
    compute_network_snr,
    compute_observed_strain,
    get_ifo_geometry,
)


@scriptify
def main(
    start: float,
    stop: float,
    shifts: List[float],
    background: Path,
    spacing: float,
    buffer: float,
    waveform_duration: float,
    cosmology: Callable,
    prior: Callable,
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_approximant: str,
    highpass: float,
    snr_threshold: float,
    ifos: List[str],
    output_fname: Path,
    log_file: Optional[Path] = None,
    verbose: bool = False,
):
    """
    Generates the waveforms for a single segment
    """

    configure_logging(log_file, verbose=verbose)

    cosmology = cosmology()
    prior, detector_frame_prior = prior(cosmology)

    injection_times = utils.calc_segment_injection_times(
        start,
        stop - max(shifts),  # TODO: should account for uneven last batch too
        spacing,
        buffer,
        waveform_duration,
    )
    n_samples = len(injection_times)
    waveform_size = int(sample_rate * waveform_duration)

    zeros = np.zeros((n_samples,))
    parameters = defaultdict(lambda: zeros.copy())
    parameters["gps_time"] = injection_times
    parameters["shift"] = np.array([shifts for _ in range(n_samples)])

    for ifo in ifos:
        empty = np.zeros((n_samples, waveform_size))
        parameters[ifo.lower()] = empty

    tensors, vertices = get_ifo_geometry(*ifos)
    df = 1 / waveform_duration
    try:
        background = next(background.iterdir())
    except StopIteration:
        raise ValueError(f"No files in background data directory {background}")
    psds = utils.load_psds(background, ifos, sample_rate=sample_rate, df=df)

    # loop until we've generated enough signals
    # with large enough snr to fill the segment,
    # keeping track of the number of signals rejected
    num_injections, idx = 0, 0
    while n_samples > 0:
        params = prior.sample(n_samples)
        waveforms = generate_gw(
            params,
            minimum_frequency,
            reference_frequency,
            sample_rate,
            waveform_duration,
            waveform_approximant,
            detector_frame_prior,
        )
        polarizations = {
            "cross": torch.Tensor(waveforms[:, 0, :]),
            "plus": torch.Tensor(waveforms[:, 1, :]),
        }

        projected = compute_observed_strain(
            torch.Tensor(params["dec"]),
            torch.Tensor(params["psi"]),
            torch.Tensor(params["ra"]),
            tensors,
            vertices,
            sample_rate,
            **polarizations,
        )
        # TODO: compute individual ifo snr so we can store that data
        snrs = compute_network_snr(projected, psds, sample_rate, highpass)
        snrs = snrs.numpy()

        # add all snrs: masking will take place in for loop below
        params["snr"] = snrs
        num_injections += len(snrs)
        mask = snrs >= snr_threshold

        num_accepted = mask.sum()
        if num_accepted == 0:
            continue

        n_samples -= num_accepted
        if n_samples < 0:
            num_accepted += n_samples
        start, stop = idx, idx + num_accepted

        for key, value in params.items():
            parameters[key][start:stop] = value[mask][:num_accepted]

        projected = projected[mask].numpy()[:num_accepted]
        for i, ifo in enumerate(ifos):
            key = ifo.lower()
            parameters[key][start:stop] = projected[:, i]
        idx += num_accepted

    parameters["sample_rate"] = sample_rate
    parameters["duration"] = waveform_duration
    parameters["num_injections"] = num_injections

    response_set = LigoResponseSet(**parameters)
    utils.io_with_blocking(response_set.write, output_fname)
    return output_fname


# until typeo update gets in just take all the same parameter as main
@scriptify
def deploy(
    start: float,
    stop: float,
    state_flag: str,
    Tb: float,
    shifts: Iterable[float],
    spacing: float,
    buffer: float,
    min_segment_length: float,
    cosmology: str,
    waveform_duration: float,
    prior: str,
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_approximant: str,
    highpass: float,
    snr_threshold: float,
    ifos: List[str],
    outdir: Path,
    datadir: Path,
    logdir: Path,
    accounting_group_user: str,
    accounting_group: str,
    request_memory: int = 6000,
    request_disk: int = 1024,
    verbose: bool = False,
    force_generation: bool = False,
) -> None:
    outdir = outdir / "timeslide_waveforms"
    writedir = datadir / "test"
    for d in [outdir, writedir, logdir]:
        d.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "timeslide_waveforms.log", verbose=verbose)

    output_fname = writedir / "waveforms.h5"
    if output_fname.exists() and not force_generation:
        logging.info(
            "Timeslide waveform file {} already exists, "
            "skipping waveform generation".format(output_fname)
        )

    # where condor info and sub files will live
    condor_dir = outdir / "condor"
    condor_dir.mkdir(exist_ok=True, parents=True)

    # query segments and calculate shifts required
    # to accumulate desired background livetime
    state_flags = [f"{ifo}:{state_flag}" for ifo in ifos]
    segments = query_segments(state_flags, start, stop, min_segment_length)
    shifts_required = utils.calc_shifts_required(segments, Tb, max(shifts))

    # create text file from which the condor job will read
    # the start, stop, and shift for each job
    parameters = "start,stop,shift\n"
    for start, stop in segments:
        for i in range(shifts_required):
            # TODO: make this more general
            shift = [i * shift for shift in shifts]
            shift = " ".join(map(str, shift))
            parameters += f"{start},{stop},{shift}\n"

    # TODO: have typeo do this CLI argument construction?
    arguments = "--start $(start) --stop $(stop) --shifts $(shift) "
    arguments += f"--background {datadir / 'train' / 'background'} "
    arguments += f"--spacing {spacing} --buffer {buffer} "
    arguments += f"--waveform-duration {waveform_duration} "
    arguments += f"--minimum-frequency {minimum_frequency} "
    arguments += f"--reference-frequency {reference_frequency} "
    arguments += f"--sample-rate {sample_rate} "
    arguments += f"--waveform-approximant {waveform_approximant} "
    arguments += f"--highpass {highpass} --snr-threshold {snr_threshold} "
    arguments += f"--ifos {' '.join(ifos)} "
    arguments += f"--prior {prior} --cosmology {cosmology} "
    arguments += f"--output-fname {outdir}/tmp-$(ProcID).h5 "
    arguments += f"--log-file {logdir}/$(ProcID).log "

    # create submit file by hand: pycondor doesn't support
    # "queue ... from" syntax
    subfile = condor.make_submit_file(
        executable="generate-timeslide-waveforms",
        name="timeslide_waveforms",
        parameters=parameters,
        arguments=arguments,
        submit_dir=condor_dir,
        accounting_group=accounting_group,
        accounting_group_user=accounting_group_user,
        clear=True,
        request_memory=request_memory,
        request_disk=request_disk,
    )
    dag_id = condor.submit(subfile)
    logging.info(f"Launching waveform generation jobs with dag id {dag_id}")
    condor.watch(dag_id, condor_dir)

    # once all jobs are done, merge the output files
    logging.info(f"Merging output files to file {output_fname}")
    utils.merge_output(outdir, output_fname)

    logging.info("Timeslide waveform generation complete")
