from typing import Callable, Optional

import numpy as np
import torch

from ml4gw.dataloading import InMemoryDataset


class BBHInMemoryDataset(InMemoryDataset):
    def __init__(
        self,
        X: np.ndarray,
        kernel_size: int,
        batch_size: int = 32,
        stride: int = 1,
        batches_per_epoch: Optional[int] = None,
        preprocessor: Optional[Callable] = None,
        coincident: bool = True,
        shuffle: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            X,
            kernel_size,
            batch_size=batch_size,
            stride=1,
            batches_per_epoch=batches_per_epoch,
            coincident=coincident,
            shuffle=shuffle,
            device=device,
        )
        self.preprocessor = preprocessor

    def __next__(self):
        X = super().__next__()
        y = torch.zeros((len(X), 1))

        if self.preprocessor is not None:
            X, y = self.preprocessor(X, y)
        return X, y
