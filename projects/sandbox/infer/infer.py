import logging
import time
from dataclasses import dataclass
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
    def __init__(self, t, x, stream_size):
        self.t = t
        self.x = x
        self.stream_size = stream_size

        self.y = np.array([])
        self.request_id = -1
        self.num_streams = len(self.x) // self.stream_size

    def update(self, y):
        self.y = np.append([self.y, y])

    def __next__(self):
        self.request_id += 1
        if self.request_id == self.num_streams:
            raise StopIteration

        start = self.request_id * self.stream_size
        stop = (self.request_id + 1) * self.stream_size
        sequence_end = self.request_id == (self.num_streams - 1)
        return self.x[start:stop], self.request_id, sequence_end

    @property
    def finished(self):
        return len(self.t) == len(self.y)


class SequenceManager:
    def __init__(self, executor, base_sequence_id):
        self.executor = executor
        self.base_sequence_id = base_sequence_id

        self.sequences = {}
        self.futures = []

    def add(self, sequence: Sequence, write_dir: Path):
        id = self.base_sequence_id + len(self.sequences)
        logging.debug(
            "Adding sequence starting at t={} for write dir {} "
            "to sequence id {}".format(sequence.t[0], write_dir, id)
        )
        self.sequences[id] = (sequence, write_dir)
        return id

    def __call__(self, y, request_id, sequence_id):
        self.sequences[sequence_id][0].update(y)
        if self.sequences[sequence_id][0].finished:
            logging.debug(f"Finished inference on sequence {sequence_id}")
            sequence, write_dir = self.sequences.pop(sequence_id)
            future = self.executor.submit(
                write_timeseries, write_dir, y=sequence.y, t=sequence.t
            )
            self.futures.append(future)

    def wait(self):
        for fname in as_completed(self.futures):
            logging.debug(f"Wrote inferred segment to file '{fname}'")
        self.futures = []


class Loader:
    def __init__(self, stride_size, batch_size, fduration):
        self.stride_size = stride_size
        self.batch_size = batch_size
        self.fduration = fduration

    @property
    def stream_size(self):
        return self.stride_size * self.batch_size

    def __call__(self, segment: Segment):
        hanford, t = segment.load("H1")
        livingston, _ = segment.load("L1")
        logging.debug(f"Loaded data from segment {segment}")

        num_streams = len(t) // self.stream_size
        t = t[: num_streams * self.stream_size : self.stride_size]

        x = np.stack([hanford, livingston])
        x = x[: num_streams * self.stream_size].astype("float32")

        sequence = Sequence(t, x, self.stream_size)
        return sequence


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
        executor = AsyncExecutor(num_workers, thread=False)

        manager = SequenceManager(executor, base_sequence_id)
        client = InferenceClient(
            "localhost:8001", model_name, model_version, callback=manager
        )
        loader = Loader(stride_size, batch_size, fduration)

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
        with client, executor, monitor:
            for shift, field in product(data_dir.iterdir(), fields):
                ts_write_dir = write_dir / shift.name / f"{field}-out"
                ts_write_dir.mkdir(parents=True, exist_ok=True)
                timeslide = TimeSlide(shift, field)

                futures = []
                for segment in timeslide.segments:
                    future = executor.submit(loader, segment)
                    callback = lambda f: manager.add(f.result(), ts_write_dir)
                    future.add_done_callback(callback)
                    futures.append(future)

                finished_sequences = []
                while True:
                    seq_ids = list(manager.sequences)
                    for seq_id in seq_ids:
                        if seq_id in finished_sequences:
                            continue

                        it, _ = manager.sequences[seq_id]
                        try:
                            x, request_id, sequence_end = next(it)
                        except StopIteration:
                            finished_sequences.append(seq_id)
                            continue

                        client.infer(
                            x,
                            request_id=request_id,
                            sequence_id=seq_id,
                            sequence_start=request_id == 0,
                            sequence_end=sequence_end,
                        )
                        if sequence_end:
                            finished_sequences.append(seq_id)
                        time.sleep(1 / inference_rate / len(manager.sequences))

                    if (
                        len(manager.sequences) == 0
                        and all([f.done() for f in futures])
                    ):
                        exc = futures[0].exception()
                        if exc is not None:
                            raise exc
                        logging.debug(f"Finished inference for {timeslide}")
                        break
                       
                manager.wait()


if __name__ == "__main__":
    main()
