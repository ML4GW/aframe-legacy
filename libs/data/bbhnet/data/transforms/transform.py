from typing import Union

import numpy as np
import torch

TENSORABLE = Union[float, np.ndarray, torch.Tensor]


def _make_tensor(
    value: TENSORABLE,
    device: torch.device,
    dtype: torch.Type = torch.float32,
):
    """
    Quick utility function for more robustly
    initializing scalar tensors
    """
    try:
        value = torch.Tensor(value)
    except TypeError:
        value = torch.Tensor([value])
    return value.type(dtype).to(device)


class Transform(torch.nn.Module):
    def fit(self, X: torch.Tensor):
        # TODO: raise NotImplementedError? Some transforms
        # may not require fitting, but should we make them
        # override this manually?
        pass

    def add_parameter(
        self,
        data: TENSORABLE,
        device: torch.device = "cpu",
        dtype: torch.Type = torch.float32,
    ) -> torch.nn.Parameter:
        value = _make_tensor(data, device, dtype)
        return torch.nn.Parameter(value, requires_grad=False)

    def set_value(
        self, parameter: torch.nn.Parameter, data: TENSORABLE
    ) -> None:
        parameter.data = _make_tensor(data, parameter.device, parameter.type())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
