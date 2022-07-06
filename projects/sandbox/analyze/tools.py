import logging
from collections import defaultdict
from concurrent.futures import FIRST_EXCEPTION, wait
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Union

import h5py
import numpy as np
from rich.progress import Progress

from bbhnet.analysis.analysis import integrate
from bbhnet.analysis.distributions import DiscreteDistribution
from bbhnet.analysis.normalizers import GaussianNormalizer
from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment
from bbhnet.parallelize import AsyncExecutor, as_completed

if TYPE_CHECKING:
    from bbhnet.analysis.distributions.distribution import Distribution


# TODO: move these functions to library?
def load_segment(segment: Segment):
    """
    Quick utility function which just wraps a Segment's
    `load` method so that we can execute it in a process
    pool since methods aren't picklable.
    """
    segment.load("out")
    return segment


def get_write_dir(
    write_dir: Path,
    norm: Optional[float],
    shift: Union[str, Segment],
    label: str,
) -> Path:
    """
    Quick utility function for getting the name of the directory
    to which to save the outputs from an analysis using a particular
    time-shift/norm-seconds combination
    """

    if isinstance(shift, Segment):
        shift = shift.shift

    write_dir = write_dir / shift / f"{label}-norm-seconds.{norm}"
    write_dir.mkdir(parents=True, exist_ok=True)
    return write_dir


def build_background(
    thread_ex: AsyncExecutor,
    process_ex: AsyncExecutor,
    pbar: Progress,
    background_segments: Iterable[Segment],
    data_dir: Path,
    write_dir: Path,
    max_tb: float,
    window_length: float = 1.0,
    norm_seconds: Optional[Iterable[float]] = None,
    num_bins: int = int(1e4),
):
    """
    For a sequence of background segments, compute a discrete
    distribution of integrated neural network outputs using
    the indicated integration window length for each of the
    normalization window lengths specified. Iterates through
    the background segments in order and tries to find as
    many time-shifts available for each segment as possible
    in the specified data directory, stopping iteration through
    segments once a maximum number of seconds of bacgkround have
    been generated.
    As a warning, there's a fair amount of asynchronous execution
    going on in this function, and it may come off a bit complex.
    Args:
        thread_ex:
            An `AsyncExecutor` that maintains a thread pool
            for writing analyzed segments in parallel with
            the analysis processes themselves.
        process_ex:
            An `AsyncExecutor` that maintains a process pool
            for loading and integrating Segments of neural
            network outputs.
        pbar:
            A `rich.progress.Progress` object for keeping
            track of the progress of each of the various
            subtasks.
        background_segments:
            The `Segment` objects to use for building a
            background distribution. `data_dir` will be
            searched for all time-shifts of each segment
            for parallel analysis. Once `max_tb` seconds
            worth of background have been generated, iteration
            through this array will be terminated, so segments
            should be ordered by some level of "importance",
            since it's likely that segments near the back of the
            array won't be analyzed for lower values of `max_tb`.
        data_dir:
            Directory containing timeslide root directories,
            which will be mined for time-shifts of each `Segment`
            in `background_segments`. If a time-shift doesn't exist
            for a given `Segment`, the time-shift is ignored.
        write_dir:
            Root directory to which to write integrated NN outputs.
            For each time-shift analyzed and normalization window
            length specified in `norm_seconds`, results will be
            written to a subdirectory
            `write_dir / "norm-seconds.{norm}" / shift`, which
            will be created if it does not exist.
        max_tb:
            The maximum number of seconds of background data
            to analyze for each value of `norm_seconds` before
            new segments to shift and analyze are no longer sought.
            However, because we use _every_ time-shift for each
            segment we iterate through, its possible that each
            background distribution will utilize slightly more
            than this value.
        window_length:
            The length of the integration window to use
            for analysis in seconds.
        norm_seconds:
            An array of normalization window lengths to use
            to standardize the integrated neural network outputs.
            (i.e. the output timeseries is the integral over the
            previous `window_length` seconds, normalized by the
            mean and standard deviation of the previous `norm`
            seconds before that, where `norm` is each value in
            `norm_seconds`). A `norm` value of `None` in the
            `norm_seconds` iterable indicates
            no normalization, and if `norm_seconds` is left as
            `None` this will be the only value used.
        num_bins:
            The number of bins to use to initialize the discrete
            distribution used to characterize the background
            distribution.
    Returns:
        A dictionary mapping each value in `norm_seconds` to
            an associated `DiscreteDistribution` characterizing
            its background distribution.
    """

    write_dir.mkdir(exist_ok=True)
    norm_seconds = norm_seconds or [norm_seconds]

    # keep track of the min and max values of each normalization
    # window's background and the corresponding filenames so
    # that we can fit a discrete distribution to it after the fact
    mins = defaultdict(lambda: float("inf"))
    maxs = defaultdict(lambda: -float("inf"))

    # keep track of all the files that we've written
    # for each normalization window size so that we
    # can iterate through them later and submit them
    # for reloading once we have our distributions initialized
    fname_futures = defaultdict(list)

    # iterate through timeshifts of our background segments
    # until we've generated enough background data.
    background_segments = iter(background_segments)
    main_task_id = pbar.add_task("[red]Building background", total=max_tb)
    while not pbar.tasks[main_task_id].finished:
        segment = next(background_segments)

        # since we're assuming here that the background
        # segments are being provided in reverse chronological
        # order (with segments closest to the event segment first),
        # exhaust all the time shifts we can of each segment before
        # going to the previous one to keep data as fresh as possible
        load_futures = {}
        for shift in data_dir.iterdir():
            try:
                shifted = segment.make_shift(shift.name)
            except ValueError:
                # this segment doesn't have a shift
                # at this value, so just move on
                continue

            # load all the timeslides up front in a separate thread
            # TODO: O(1GB) memory means segment.length * N ~O(4M),
            # so for ~O(10k) long segments this means this should
            # be fine as long as N ~ O(100). Worth doing a check for?
            future = process_ex.submit(load_segment, shifted)
            load_futures[shift.name] = [future]

        # create progress bar tasks for each one
        # of the subprocesses involved for analyzing
        # this set of timeslides
        load_task_id = pbar.add_task(
            f"[cyan]Loading {len(load_futures)} {segment.length}s timeslides",
            total=len(load_futures),
        )
        analyze_task_id = pbar.add_task(
            "[yelllow]Integrating timeslides",
            total=len(load_futures) * len(norm_seconds),
        )
        write_task_id = pbar.add_task(
            "[green]Writing integrated timeslides",
            total=len(load_futures) * len(norm_seconds),
        )

        # now once each segment is loaded, submit a job
        # to our process pool to integrate it using each
        # one of the specified normalization periods
        integration_futures = {}
        sample_rate = None
        for shift, seg in as_completed(load_futures):
            # get the sample rate of the NN output timeseries
            # dynamically from the first timeseries we load,
            # since we'll need it to initialize our normalizers
            if sample_rate is None:
                t = seg._cache["t"]
                sample_rate = 1 / (t[1] - t[0])

            for norm in norm_seconds:
                # build a normalizer for the given normalization window length
                if norm is not None:
                    normalizer = GaussianNormalizer(norm * sample_rate)
                else:
                    normalizer = None

                # submit the integration job and have it update the
                # corresponding progress bar task once it completes
                future = process_ex.submit(
                    integrate,
                    seg,
                    kernel_length=1.0,
                    window_length=window_length,
                    normalizer=normalizer,
                )
                future.add_done_callback(
                    lambda f: pbar.update(analyze_task_id, advance=1)
                )
                integration_futures[(norm, shift)] = [future]

            # advance the task keeping track of how many files
            # we've loaded by one
            pbar.update(load_task_id, advance=1)

        # make sure we have the expected number of jobs submitted
        if len(integration_futures) < (len(norm_seconds) * len(load_futures)):
            raise ValueError(
                "Expected {} integration jobs submitted, "
                "but only found {}".format(
                    len(norm_seconds) * len(load_futures),
                    len(integration_futures),
                )
            )

        # as the integration jobs come back, write their
        # results using our thread pool and record the
        # min and max values for our discrete distribution
        segment_futures = []
        for (norm, shift), (t, y, integrated) in as_completed(
            integration_futures
        ):
            # submit the writing job to our thread pool and
            # use a callback to keep track of all the filenames
            # for a given normalization window
            shift_dir = get_write_dir(write_dir, norm, shift, "background")
            future = thread_ex.submit(
                write_timeseries,
                shift_dir,
                t=t,
                y=y,
                integrated=integrated,
            )
            future.add_done_callback(
                lambda f: pbar.update(write_task_id, advance=1)
            )
            fname_futures[norm].append(future)
            segment_futures.append(future)

            # keep track of the max and min values for each norm
            mins[norm] = min(mins[norm], integrated.min())
            maxs[norm] = max(maxs[norm], integrated.max())

        # wait for all the writing to finish before we
        # move on so that we don't overload our processes
        wait(segment_futures, return_when=FIRST_EXCEPTION)
        pbar.update(main_task_id, advance=len(load_futures) * segment.length)

    # now that we've analyzed enough background data,
    # we'll initialize background distributions using
    # the min and max bounds we found during analysis
    # and then load everything back in to bin them
    # within these bounds
    Tb = pbar.tasks[main_task_id].completed
    logging.info(f"Accumulated {Tb}s of background matched filter outputs.")

    # submit a bunch of jobs for loading these integrated
    # segments back in for discretization
    load_futures = defaultdict(list)
    for norm, fname in as_completed(fname_futures):
        future = process_ex.submit(load_segment, Segment(fname))
        load_futures[norm].append(future)

    # create a task for each one of the normalization windows
    # tracking how far along the distribution fit is
    fit_task_ids = {}
    for norm in norm_seconds:
        norm_name = f"{norm}s" if norm is not None else "empty"
        task_id = pbar.add_task(
            "[purple]Fitting background using {} normalization window".format(
                norm_name
            ),
            total=len(load_futures[norm]),
        )
        fit_task_ids[norm] = task_id

    # now discretized the analyzed segments as they're loaded back in
    backgrounds = {}
    for norm, segment in as_completed(load_futures):
        try:
            # if we already have a background distribution
            # for this event, grab it and fit it with a
            # "warm start" aka don't ditch the existing histogram
            background = backgrounds[norm]
            warm_start = True
        except KeyError:
            # otherwise create a new distribution
            # and fit it from scratch
            mn, mx = mins[norm], maxs[norm]
            background = DiscreteDistribution("integrated", mn, mx, num_bins)
            backgrounds[norm] = background
            warm_start = False

        # fit the distribution to the new data and then
        # update the corresponding task tracker
        background.fit(segment, warm_start=warm_start)
        pbar.update(fit_task_ids[norm], advance=1)

    return backgrounds


def analyze_injections(
    process_ex: AsyncExecutor,
    thread_ex: AsyncExecutor,
    data_dir: Path,
    write_dir: Path,
    results_dir: Path,
    backgrounds: Dict[str, "Distribution"],
    event_times: Iterable[float],
    injection_segments: Iterable[Segment],
    window_length: float = 1.0,
):
    """Analyzes a set of events injected on top of timeslides

    data_dir:
            Directory containing timeslide root directories,
            which will be mined for time-shifts of each `Segment`
            in `background_segments`. If a time-shift doesn't exist
            for a given `Segment`, the time-shift is ignored.
    """

    # for each normalization window
    # in backgrounds dictionary
    # create normalizer
    for norm, background in backgrounds.items():
        master_fars, master_latencies, master_event_times = [], [], []
        if norm is not None:
            normalizer = GaussianNormalizer(norm)
        else:
            normalizer = None

        # loop over injection segments
        for segment in injection_segments:

            # restrict event times of interest
            # to those that are in this segment and
            # norm seconds away from start
            segment_event_times = [
                time
                for time in event_times
                if time < segment.tf and (time - norm) > segment.t0
            ]

            # integrate this injection segment
            # for all timeshits up front;
            # the integrate function
            # takes care of segment loading
            integrate_futures = {}
            for shift in data_dir.iterdir():
                try:
                    shifted = segment.make_shift(shift.name)
                except ValueError:
                    # this segment doesn't have a shift
                    # at this value, so just move on
                    continue

                # submit integration job
                future = process_ex.submit(
                    integrate,
                    shifted,
                    kernel_length=1,
                    window_length=window_length,
                    normalizer=normalizer,
                )

                integrate_futures[shift.name] = [future]

            # as the integration jobs come back
            # submit write jobs
            write_futures = []
            for shift, (t, y, integrated) in as_completed(integrate_futures):
                future = thread_ex.submit(
                    write_timeseries,
                    get_write_dir(write_dir, norm, shift, "injection"),
                    t=t,
                    y=y,
                    integrated=integrated,
                )

                write_futures.append(future)

            # as the write jobs complete,
            # create Segment objects for them
            # and characterize all of the events in
            # this Segment
            for fname in as_completed(write_futures):
                # create a segment and add the existing data to
                # its cache so that we don't try to load it again
                segment = Segment(fname)
                segment._cache = {"t": t, "integrated": integrated}

                # characterize all of the events in this segment
                fars, latencies = background.characterize_events(
                    segment,
                    segment_event_times,
                    window_length=window_length,
                    metric="far",
                )

                master_fars.append(fars)
                master_latencies.append(latencies)
                master_event_times.append(segment_event_times)

        logging.info(np.shape(master_fars))
        master_fars = np.stack(master_fars)
        master_latencies = np.stack(master_latencies)
        master_event_times = np.stack(master_event_times)

        with h5py.File(results_dir / f"injections-{norm}.h5", "w") as f:
            f.create_dataset("fars", data=master_fars)

            f.create_dataset("latencies", data=master_latencies)
            f.create_dataset("event_times", data=master_event_times)
