from pathlib import Path
from typing import Dict, List, Optional

from bokeh.server.server import Server

from bbhnet.logging import configure_logging
from hermes.typeo import typeo

from .app import VizApp
from .vetoes import VetoeParser


@typeo
def main(
    ifos: List[str],
    veto_definer_file: Path,
    gate_paths: Dict[str, Path],
    timeslides_results_dir: Path,
    timeslides_strain_dir: Path,
    train_data_dir: Path,
    start: float,
    stop: float,
    sample_rate: float,
    fduration: float,
    valid_frac: float,
    port: int = 5005,
    logdir: Optional[Path] = None,
    verbose: bool = False,
) -> None:

    configure_logging(logdir / "vizapp.log", verbose)

    vetoe_parser = VetoeParser(
        veto_definer_file,
        gate_paths,
        start,
        stop,
        ifos,
    )

    bkapp = VizApp(
        timeslides_results_dir=timeslides_results_dir,
        timeslides_strain_dir=timeslides_strain_dir,
        train_data_dir=train_data_dir,
        vetoe_parser=vetoe_parser,
        ifos=ifos,
        sample_rate=sample_rate,
        fduration=fduration,
        valid_frac=valid_frac,
    )

    server = Server({"/": bkapp}, num_procs=1, port=port, address="0.0.0.0")
    server.start()
    server.run_until_shutdown()


if __name__ == "__main__":
    main()
