import logging
from pathlib import Path
from typing import List

from datagen.scripts.background import validate_segments

from aframe.deploy import condor
from mldatafind import query_segments


def deploy_background(
    start: float,
    stop: float,
    state_flag: str,
    channel: str,
    sample_rate: float,
    ifos: List[str],
    min_segment_length: float,
    max_segment_length: float,
    outdir: Path,
    accounting_group: str,
    accounting_group_user: str,
    request_memory: int = 32768,
    request_disk: int = 1024,
    watch: bool = False,
):
    state_flags = [f"{ifo}:{state_flag}" for ifo in ifos]
    segments = query_segments(
        state_flags, start, stop, min_duration=min_segment_length
    )
    segments = validate_segments(
        segments,
        train_start=0,  # dummy value to skip training
        train_stop=start,
        test_stop=stop,
        minimum_train_length=0,
        minimum_test_length=min_segment_length,
        max_segment_length=max_segment_length,
        datadir=outdir,
        force_generation=True,
        ifos=ifos,
        sample_rate=sample_rate,
    )
    # create text file from which the condor job will read
    # the start, stop, and shift for each job
    parameters = "start,stop,writepath\n"
    for start, stop, writepath in segments:
        parameters += f"{start},{stop},{writepath}\n"

    arguments = "--start $(start) --stop $(stop) "
    arguments += "--writepath $(writepath) "
    arguments += f"--channel {channel} --sample-rate {sample_rate} "
    arguments += f"--ifos {' '.join(ifos)} "

    condordir = outdir / "condor" / "background"
    condordir.mkdir(exist_ok=True, parents=True)

    kwargs = {"+InitialRequestMemory": request_memory}
    subfile = condor.make_submit_file(
        executable="generate-background",
        name="generate_background",
        parameters=parameters,
        arguments=arguments,
        submit_dir=condordir,
        accounting_group=accounting_group,
        accounting_group_user=accounting_group_user,
        clear=True,
        request_disk=request_disk,
        # stolen from pyomicron: allows dynamic updating of memory
        request_memory=f"ifthenelse(isUndefined(MemoryUsage), {request_memory}, int(3*MemoryUsage))",  # noqa
        periodic_release="(HoldReasonCode =?= 26 || HoldReasonCode =?= 34) && (JobStatus == 5)",  # noqa
        periodic_remove="(JobStatus == 1) && MemoryUsage >= 7G",
        use_x509userproxy=True,
        **kwargs,
    )
    dag_id = condor.submit(subfile)
    logging.info(f"Launching background generation jobs with dag id {dag_id}")
    if watch:
        condor.watch(dag_id, condordir, held=True)
    logging.info("Completed background generation jobs")
