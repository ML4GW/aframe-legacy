from dataclasses import dataclass
from pathlib import Path


@dataclass
class Deployment:
    root: Path

    def __post_init__(self):
        for d in ["log", "train", "data", "omicron", "repository"]:
            dirname = getattr(self, f"{d}_directory")
            dirname.mkdir(exist_ok=True, parents=True)

    @property
    def log_directory(self):
        return self.root / "logs"

    @property
    def train_directory(self):
        return self.root / "train"

    @property
    def data_directory(self):
        return self.root / "data"

    @property
    def omicron_directory(self):
        return self.root / "omicron"

    @property
    def repository_directory(self):
        return self.root / "model_repo"
