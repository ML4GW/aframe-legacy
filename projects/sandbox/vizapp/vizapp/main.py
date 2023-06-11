import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from bokeh.server.server import Server

from aframe.architectures import architecturize
from aframe.logging import configure_logging

from .app import VizApp
from .vetoes import VetoParser

logging.getLogger("urllib3").setLevel(logging.WARNING)


def _normalize_path(path: Path):
    if not path.is_absolute():
        return Path(__file__).resolve().parent / path
    return path


@architecturize
def main(
    architecture: Callable,
    basedir: Path,
    datadir: Path,
    veto_definer_file: Path,
    gate_paths: Dict[str, Path],
    ifos: List[str],
    cosmology: Callable,
    source_prior: Callable,
    start: float,
    stop: float,
    sample_rate: float,
    fduration: float,
    valid_frac: float,
    device: str = "cpu",
    port: int = 5005,
    logdir: Optional[Path] = None,
    verbose: bool = False,
) -> None:

    logfile = logdir / "vizapp.log" if logdir is not None else None
    configure_logging(logfile, verbose)

    model = architecture(len(ifos))
    model.to(device)

    weights = basedir / "training" / "weights.pt"
    model.load_state_dict(
        torch.load(weights, map_location=torch.device(device))
    )

    veto_definer_file = _normalize_path(veto_definer_file)
    for ifo in ifos:
        gate_paths[ifo] = _normalize_path(gate_paths[ifo])

    veto_parser = VetoParser(
        veto_definer_file,
        gate_paths,
        start,
        stop,
        ifos,
    )

    cosmology = cosmology()
    source_prior, _ = source_prior()
    bkapp = VizApp(
        model=model,
        base_directory=basedir,
        data_directory=datadir,
        cosmology=cosmology,
        source_prior=source_prior,
        ifos=ifos,
        sample_rate=sample_rate,
        fduration=fduration,
        valid_frac=valid_frac,
        veto_parser=veto_parser,
    )

    server = Server({"/": bkapp}, num_procs=1, port=port, address="0.0.0.0")
    server.start()
    server.run_until_shutdown()


if __name__ == "__main__":
    main()
