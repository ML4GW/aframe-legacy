import logging
import time
from itertools import product
from pathlib import Path
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

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


class Sequence:
    def __init__(self, t, write_dir):
        self.t = t
        self.y = np.array([])
        self.write_dir = write_dir

    def update(self, y):
        self.y = np.append([self.y, y])

    @property
    def finished(self):
        return len(self.t) == len(self.y)


class SequenceManager:
    def __init__(self, executor):
        self.executor = executor
        self.sequences = {}
        self.futures = []

    def add(self, sequence: Sequence, seq_id: int):
        logging.debug(
            "Managing sequence with t0={} as id {}".format(
                sequence.t[0], seq_id
            )
        )
        self.sequences[seq_id] = sequence

    def __call__(self, y, request_id, sequence_id):
        self.sequences[sequence_id].update(y)

        if self.sequences[sequence_id].finished:
            logging.debug(f"Finished inference on sequence {sequence_id}")
            sequence, write_dir = self.sequences.pop(sequence_id)
            future = self.executor.submit(
                write_timeseries,
                sequence.write_dir,
                y=sequence.y,
                t=sequence.t,
            )
            self.futures.append(future)

    def wait(self):
        for fname in as_completed(self.futures):
            logging.debug(f"Wrote inferred segment to file '{fname}'")
        self.futures = []


def load(
    segment: Segment,
    write_dir: Path,
    stride_size: int,
    batch_size: int,
    fduration: Optional[float] = None,
):
    hanford, t = segment.load("H1")
    livingston, _ = segment.load("L1")
    logging.debug(f"Loaded data from segment {segment}")

    stream_size = stride_size * batch_size
    num_streams = len(t) // stream_size
    t = t[: num_streams * stream_size : stride_size]

    x = np.stack([hanford, livingston])
    x = x[: num_streams * stream_size].astype("float32")

    sequence = Sequence(t, write_dir)
    return sequence, x


def infer(
    x: np.ndarray,
    stream_size: int,
    sequence_id: int,
    client: InferenceClient,
    inference_rate: float,
):
    num_streams = len(x) // stream_size
    x = np.split(x, num_streams)

    logging.debug(f"Beginning inference on sequence {sequence_id}")
    for i, update in enumerate(x):
        client.infer(
            update,
            request_id=i,
            sequence_id=sequence_id,
            sequence_start=i == 0,
            sequence_end=i == (len(x) - 1),
        )
        time.sleep(1 / inference_rate)


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
        # now build a client to connect to the inference service
        # create a process pool that we'll use to perform
        # read/writes of timeseries in parallel
        io_pool = AsyncExecutor(num_workers, thread=False)
        infer_pool = AsyncExecutor(num_workers, thread=True)
        callback = SequenceManager(io_pool)
        client = InferenceClient(
            "localhost:8001", model_name, model_version, callback=callback
        )

        monitor = ServerMonitor(
            model_name=model_name,
            ips="localhost",
            filename=log_file.parent / "server-stats.csv",
            model_version=model_version,
            name="monitor",
        )

        # now enter a context which will:
        # - for the client, start a streaming connection with
        #       with the inference service and launch a separate
        #       process for inference
        # - for the executor, launch the process pool
        with client, io_pool, infer_pool, monitor:
            for shift, field in product(data_dir.iterdir(), fields):
                ts_write_dir = write_dir / shift.name / f"{field}-out"
                ts_write_dir.mkdir(parents=True, exist_ok=True)
                timeslide = TimeSlide(shift, field)

                futures = []
                for i, segment in enumerate(timeslide.segments):
                    future = io_pool.submit(
                        load,
                        segment,
                        ts_write_dir,
                        stride_size,
                        batch_size,
                        fduration,
                    )

                    # add it to our manager callback that appends
                    # server responses and submits finished sequences
                    # for writing
                    seq_id = base_sequence_id + i
                    future.add_done_callback(
                        lambda f: callback.add(f.result()[0], seq_id)
                    )

                    # submit an inference job with it to our infer_pool
                    future.add_done_callback(
                        lambda f: infer_pool.submit(
                            infer,
                            f.result()[1],
                            stride_size * batch_size,
                            seq_id,
                            client,
                            inference_rate,
                        )
                    )
                    futures.append(future)

                # wait until all the infer submissions are done
                _ = [i for i in as_completed(futures)]

                # now wait until all the inferences are completed
                callback.wait()


if __name__ == "__main__":
    main()
