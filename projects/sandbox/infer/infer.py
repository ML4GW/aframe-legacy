import time
from pathlib import Path
from typing import Iterable

import numpy as np
from hermes.stillwater import InferenceClient
from hermes.stillwater.utils import Package
from hermes.typeo import typeo

from bbhnet.io.timeslides import Segment, TimeSlide
from bbhnet.parallelize import AsyncExecutor
from tritonserve import serve


def load(segment: Segment):
    hanford, t = segment.load("hanford")
    livingston, _ = segment.load("livingston")
    return np.stack([hanford, livingston]), t, segment.fnames


def stream_data(
    x: np.ndarray, stream_size: int, sequence_id: int, client: InferenceClient
):
    # TODO: make more robust to when this does not evenly divide
    num_streams = x.shape[-1] // stream_size
    for i in range(num_streams):
        stream = x[i * stream_size : (i + 1) * stream_size]
        package = Package(
            x=stream,
            t0=time.time(),
            request_id=i,
            sequence_id=sequence_id,
            sequence_start=i == 0,
            sequence_end=(i + 1) == num_streams,
        )
        client.in_q.put(package)


def infer(
    client: InferenceClient,
    executor: AsyncExecutor,
    timeslides: Iterable[TimeSlide],
    stream_size: int,
    base_sequence_id: int,
):
    records = {}
    for timeslide in timeslides:
        data_it = executor.imap(load, timeslide.segments)
        for i, (x, t, fnames) in enumerate(data_it):
            sequence_id = base_sequence_id + i
            records[sequence_id] = (t, fnames)
            stream_data(x, stream_size, sequence_id, client)


@typeo
def main(
    model_repo_dir: Path,
    data_dir: Path,
    field: str,
    sample_rate: float,
    inference_sampling_rate: float,
    num_workers: int,
    base_sequence_id: int = 1001,
):
    stream_size = int(sample_rate // inference_sampling_rate)
    with serve(model_repo_dir, wait=True):
        client = InferenceClient("localhost:8001")
        executor = AsyncExecutor(num_workers, thread=False)
        timeslides = [TimeSlide(i, field) for i in data_dir.iterdir()]
        with client, executor:
            infer(client, executor, timeslides, stream_size, base_sequence_id)


if __name__ == "__main__":
    main()
