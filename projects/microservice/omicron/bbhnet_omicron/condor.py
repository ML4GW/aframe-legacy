import re
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Union

# TODO: a lot of this is copied from bbhnet.deploy
# use that library once it's merged

# logic largely lifted from https://git.ligo.org/detchar/ligo-omicron/
SUBFILE = """
# Condor submit file for aLIGO Omicron Online processing

universe = {universe}
executable = {executable}
arguments = " {arguments} "
accounting_group = {accounting_group}
accounting_group_user = {accounting_user}
batch_name = " omicron crondor - {group} "

environment = "OMICRON_HOME={archivedir}"
request_memory = 4000
request_disk = 500MB

+OmicronManager = "{group}"

output = {logdir}/omicron-online-{group}.out
error = {logdir}/omicron-online-{group}.err
log = {logdir}/omicron-online-{group}.log

on_exit_remove = false
periodic_hold = false
cron_minute = {offset}-59/{runevery}
cron_hour = *
cron_day_of_month = *
cron_month = *
cron_day_of_week = *
cron_window = 120
cron_prep_time = {preptime}

getenv = True

queue
"""


def make_submit_file(
    arguments: str,
    runevery: int,
    offset: int,
    preptime: int,
    group: str,
    logdir: Path,
    archivedir: Path,
    accounting_group: str,
    accounting_user: str,
    universe: str,
    executable: str,
    fname: Path,
):
    subfile = dedent(
        SUBFILE.format(
            arguments=arguments,
            runevery=runevery,
            offset=offset,
            preptime=preptime,
            group=group,
            logdir=logdir,
            accounting_group=accounting_group,
            accounting_user=accounting_user,
            universe=universe,
            executable=executable,
            archivedir=archivedir,
        )
    )

    with open(fname, "w") as f:
        f.write(subfile)

    return fname


def get_executable(name: str) -> str:
    ex = shutil.which(name)
    if ex is None:
        raise ValueError(f"No executable {name}")
    return str(ex)


def submit(sub_file: Union[str, Path]) -> str:
    condor_submit = get_executable("condor_submit")
    cmd = [condor_submit, str(sub_file)]
    out = subprocess.check_output(cmd, text=True)

    # re for extracting cluster id from condor_submit output
    # stolen from pyomicron:
    # https://github.com/ML4GW/pyomicron/blob/master/omicron/condor.py
    dag_id = re.search(r"(?<=submitted\sto\scluster )[0-9]+", out).group(0)

    return dag_id


def check_failed(submit_dir: Path):
    log_dir = submit_dir / "logs"
    failed_jobs, total_jobs = [], 0
    for f in log_dir.glob("*.log"):
        log = f.read_text()
        for line in log.splitlines()[-5:]:
            line = line.strip()
            if line.startswith("Job terminated") and not line.endswith(
                "exit-code 0."
            ):
                err_file = log_dir / (f.stem + ".err")
                failed_jobs.append(str(err_file))
                break
        total_jobs += 1

    if failed_jobs:
        err_files = "\n".join(failed_jobs[:10])
        if len(failed_jobs) > 10:
            err_files += "\n..."
        raise RuntimeError(
            "{}/{} jobs failed, consult error files:\n{}".format(
                len(failed_jobs), total_jobs, err_files
            )
        )


def watch(dag_id: str, submit_dir: Path):
    cwq = get_executable("condor_watch_q")
    subprocess.check_call(
        [
            cwq,
            "-exit",
            "all,done,0",
            "-exit",
            "any,held,1",
            "-clusters",
            dag_id,
        ]
    )
    check_failed(submit_dir)
