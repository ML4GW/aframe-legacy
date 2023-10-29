import shutil
from pathlib import Path

from plots.sensitive_volume import main as sensitive_volume
from typeo import scriptify

from aframe.deploy import condor
from aframe.logging import configure_logging


@scriptify
def deploy(
    basedir: Path,
    accounting_group: str,
    accounting_group_user: str,
    verbose: bool = False,
):

    configure_logging(verbose=verbose)
    condor_dir = basedir / "condor"
    condor_dir.mkdir(exist_ok=True, parents=True)
    # run sv calculation for each retrained model, and original model
    intervals = [
        x for x in basedir.iterdir() if x.is_dir() and x.name != "condor"
    ]

    parameters = "rundir,rejected_params\n"
    for interval in intervals:
        for analysis in ["original", "retrained"]:
            rejected_params = (
                interval / "data" / "test" / "rejected-parameters.h5"
            )
            analysis_dir = interval / analysis
            parameters += f"{analysis_dir},{rejected_params}\n"

    arguments = "--rundir $(rundir) --rejected-params $(rejected_params)"
    submit_file = condor.make_submit_file(
        shutil.which("sv-over-time"),
        name="sv-over-time",
        submit_dir=condor_dir,
        arguments=arguments,
        parameters=parameters,
        accounting_group=accounting_group,
        accounting_group_user=accounting_group_user,
        clear=True,
        request_memory="4GB",
        request_disk="1GB",
        request_cpus=4,
    )

    dag_id = condor.submit(submit_file)
    condor.watch(dag_id, condor_dir)


@scriptify
def main(rundir: Path, rejected_params: Path):

    background_file = rundir / "infer" / "background.h5"
    foreground_file = rundir / "infer" / "foreground.h5"
    output_dir = rundir / "results"
    log_file = rundir / "results" / "sensitive_volume.log"

    return sensitive_volume(
        background_file,
        foreground_file,
        rejected_params,
        output_dir,
        log_file,
    )


if __name__ == "__main__":
    main()
