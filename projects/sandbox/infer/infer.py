import logging
import time
from itertools import product
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional

import numpy as np

from bbhnet.io.h5 import write_timeseries
from bbhnet.io.timeslides import Segment, TimeSlide
from bbhnet.logging import configure_logging
from bbhnet.parallelize import AsyncExecutor, as_completed
from hermes.aeriel.client import InferenceClient
from hermes.aeriel.serve import serve
from hermes.stillwater import ServerMonitor
from hermes.typeo import typeo

# turn off debugging messages from request libraries
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


class Sequence:
    def __init__(self, t, x, write_dir, batch_size):
        self.t = t
        self.x = x
        self.y = np.zeros_like(t)
        self.batch_size = batch_size
        self.write_dir = write_dir

        self._last_seen = -1

    def update(self, y, request_id, sequence_id):
        if request_id > (self._last_seen + 1):
            logging.warning(
                "Dropped response for request {} for sequence {}".format(
                    self._last_seen + 1, sequence_id
                )
            )
            self._last_seen = request_id
        elif request_id < self._last_seen:
            logging.warning(
                "Request {} from sequence {} came in late".format(
                    request_id, sequence_id
                )
            )
        else:
            self._last_seen = request_id

        start = request_id * self.batch_size
        stop = (request_id + 1) * self.batch_size
        self.y[start:stop] = y[:, 0]

    @property
    def finished(self):
        return self._last_seen == (len(self.x) - 1)


class SequenceManager:
    def __init__(self, executor, base_sequence_id):
        self.executor = executor
        self.base_sequence_id = base_sequence_id

        self.lock = Lock()
        self.sequences = {}
        self.futures = []

    def add(self, sequence: Sequence):
        with self.lock:
            for i in range(len(self.sequences) + 1):
                seq_id = self.base_sequence_id + i
                if seq_id not in self.sequences:
                    break
            self.sequences[seq_id] = sequence

        logging.info(
            "Managing sequence with t0={} as id {}".format(
                sequence.t[0], seq_id
            )
        )
        return seq_id

    def __call__(self, y, request_id, sequence_id):
        self.sequences[sequence_id].update(y, request_id, sequence_id)

        if self.sequences[sequence_id].finished:
            logging.info(f"Finished inference on sequence {sequence_id}")
            with self.lock:
                sequence = self.sequences.pop(sequence_id)

            future = self.executor.submit(
                write_timeseries,
                sequence.write_dir,
                y=sequence.y,
                t=sequence.t,
            )
            future.add_done_callback(
                lambda f: logging.info(
                    "Wrote inferred sequence {} to file {}".format(
                        sequence_id, f.result()
                    )
                )
            )

            with self.lock:
                self.futures.append(future)
            return future


def load(
    segment: Segment,
    write_dir: Path,
    stride_size: int,
    batch_size: int,
    fduration: Optional[float] = None,
):
    hanford, t = segment.load("H1")
    livingston, _ = segment.load("L1")
    logging.info(f"Loaded data from segment {segment}")

    stream_size = stride_size * batch_size
    num_streams = len(t) // stream_size
    t = t[: num_streams * stream_size : stride_size]

    x = np.stack([hanford, livingston])
    x = x[:, : num_streams * stream_size].astype("float32")
    x = np.split(x, num_streams, axis=-1)

    sequence = Sequence(t, x, write_dir, batch_size)
    return sequence


def infer(
    sequence: Sequence,
    stream_size: int,
    manager: SequenceManager,
    client: InferenceClient,
    inference_rate: float,
):
    sequence_id = manager.add(sequence)
    logging.debug(f"Beginning inference on sequence {sequence_id}")
    for i, update in enumerate(sequence.x):
        logging.debug(
            "Submitting request {} for sequence {}".format(i, sequence_id)
        )
        client.infer(
            update,
            request_id=i,
            sequence_id=sequence_id,
            sequence_start=i == 0,
            sequence_end=i == (len(sequence.x) - 1),
        )
        time.sleep(1 / inference_rate)

        # let the first request complete before we
        # send anymore to make sure that the request
        # with sequence_start=True gets there first
        if i == 0:
            while sequence._last_seen < 0:
                time.sleep(1e-3)


@typeo
def main(
    model_repo_dir: str,
    model_name: str,
    data_dir: Path,
    write_dir: Path,
    fields: Iterable[str],
    sample_rate: float,
    inference_sampling_rate: float,
    inference_rate: float,
    batch_size: int,
    num_workers: int,
    model_version: int = -1,
    base_sequence_id: int = 1001,
    log_file: Optional[Path] = None,
    fduration: Optional[float] = None,
    verbose: bool = False,
):
    configure_logging(log_file, verbose)
    stride_size = int(sample_rate // inference_sampling_rate)

    if log_file is not None:
        server_log_file = log_file.parent / "server.log"
    else:
        server_log_file = None

    # spin up a triton server and don't move on until it's ready
    with serve(model_repo_dir, wait=True, log_file=server_log_file):
        # do reading and writing of segments using multiprocessing
        io_pool = AsyncExecutor(num_workers, thread=False)

        # submit inference requests using threads so that we
        # can leverage all of our snapshotter states on the server
        infer_pool = AsyncExecutor(6, thread=True)

        # create a callback which will concatenate the server
        # responses for each parallel sequence we're working
        # on and will submit them to be written once inference
        # has been completed on that sequence
        callback = SequenceManager(io_pool, base_sequence_id)

        # create a client connection to the server and give
        # it the callback to be executed once responses come in
        client = InferenceClient(
            "localhost:8001", model_name, model_version, callback=callback
        )

        # create a monitor which will recored per-model
        # inference stats for profiling purposes
        monitor = ServerMonitor(
            model_name=model_name,
            ips="localhost",
            filename=log_file.parent / "server-stats.csv",
            model_version=model_version,
            name="monitor",
        )

        # enter all of these objects as contexts
        with client, io_pool, infer_pool, monitor:
            for shift, field in product(data_dir.iterdir(), fields):
                ts_write_dir = write_dir / shift.name / f"{field}-out"
                ts_write_dir.mkdir(parents=True, exist_ok=True)
                timeslide = TimeSlide(shift, field)

                futures = []
                for segment in timeslide.segments:
                    future = io_pool.submit(
                        load,
                        segment,
                        ts_write_dir,
                        stride_size,
                        batch_size,
                        fduration,
                    )

                    # submit an inference job with it to our infer_pool
                    future.add_done_callback(
                        lambda f: infer_pool.submit(
                            infer,
                            f.result(),
                            stride_size * batch_size,
                            callback,
                            client,
                            inference_rate,
                        )
                    )
                    futures.append(future)

                # wait until all the infer submissions are done
                while len(futures) < (2 * len(timeslide.segments)):
                    result = client.get()
                    if result is not None:
                        futures.append(result)
                    time.sleep(0.01)
                _ = [i for i in as_completed(futures)]


if __name__ == "__main__":
    main()
