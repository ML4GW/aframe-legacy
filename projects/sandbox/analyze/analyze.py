from pathlib import Path
from typing import List, Optional

import h5py
from rich.progress import Progress
from tools import analyze_injections, build_background

from bbhnet.io.timeslides import TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import AsyncExecutor
from hermes.typeo import typeo


@typeo
def main(
    data_dir: Path,
    write_dir: Path,
    results_dir: Path,
    window_length: float = 1.0,
    norm_seconds: Optional[List[float]] = None,
    max_tb: Optional[float] = None,
    num_bins: int = 10000,
    force: bool = False,
    log_file: Optional[str] = None,
    verbose: bool = False,
):
    """Analyze injections in a directory of timeslides

    Iterate through a directory of timeslides analyzing known
    injections for false alarm rates in units of yrs$^{-1}$ as
    a function of the time after the event trigger times enters
    the neural network's input kernel. For each event and normalization
    period specified by `norm_seconds`, use time- shifted data from
    segments _before_ the event's segment tobuild up a background
    distribution of the output of matched filters of length `window_length`,
    normalized by the mean and standard deviation of the previous
    `norm_seconds` worth of data, until the effective time analyzed
    is equal to `max_tb`.
    The results of this analysis will be written to two csv files,
    one of which will contain the latency and false alaram rates
    for each of the events and normalization windows, and the other
    of which will contain the bins and counts for the background
    distributions used to calculate each of these false alarm rates.
    Args:
        data_dir: Path to directory containing timeslides and injections
        write_dir: Path to directory to which to write matched filter outputs
        results_dir:
            Path to directory to which to write analysis logs and
            summary csvs for analyzed events and their corresponding
            background distributions.
        window_length:
            Length of time, in seconds, over which to average
            neural network outputs for matched filter analysis
        norm_seconds:
            Length of time, in seconds, over which to compute a moving
            "background" used to normalize the averaged neural network
            outputs. More specifically, the matched filter output at each
            point in time will be the average over the last `window_length`
            seconds, normalized by the mean and standard deviation of the
            previous `norm_seconds` seconds. If left as `None`, no
            normalization will be performed. Otherwise, should be specified
            as an iterable to compute multiple different normalization values
            for each event.
        max_tb:
            The maximum number of time-shifted background data to analyze
            per event, in seconds
        num_bins:
            The number of bins to use in building up the discrete background
            distribution
        force:
            Flag indicating whether to force an event analysis to re-run
            if its data already exists in the summary files written to
            `results_dir`.
        log_file:
            A filename to write logs to. If left as `None`, logs will only
            be printed to stdout
        verbose:
            Flag indicating whether to log at level `INFO` (if set)
            or `DEBUG` (if not set)
    """

    results_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(results_dir / log_file, verbose)

    # initiate process and thread pools
    thread_ex = AsyncExecutor(4, thread=True)
    process_ex = AsyncExecutor(4, thread=False)

    # organize background and injection timeslides into segments
    background_segments = TimeSlide(
        data_dir / "dt-0.0-0.0", field="background-out"
    ).segments
    injection_segments = TimeSlide(
        data_dir / "dt-0.0-0.0", field="injection-out"
    ).segments

    # get event times from injection timeslide
    # since these event times refer to the
    # unshifted interferometer, they will be the same
    # across timeslides
    event_params_file = data_dir / "dt-0.0-0.0" / "injection" / "params.h5"
    with h5py.File(event_params_file) as f:
        event_times = f["geocent_time"][()] + 2

    with thread_ex, process_ex:
        # build background distributions
        # for all timeslides for various
        # normalization lengths
        with Progress() as pbar:
            backgrounds, sample_rate = build_background(
                thread_ex,
                process_ex,
                pbar,
                background_segments,
                data_dir,
                write_dir,
                max_tb,
                window_length,
                norm_seconds,
                num_bins,
            )

        # analyze all injection events
        analyze_injections(
            process_ex,
            thread_ex,
            data_dir,
            write_dir,
            results_dir,
            backgrounds,
            event_times,
            injection_segments,
            sample_rate,
            4,
        )


if __name__ == "__main__":
    main()
