# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn


def get_device(device="auto"):
    """
    Get the device for training and inference.

    Parameters
    ----------
    device : str
        The device to use.
        If device is "auto", it will use the first available device from CUDA, MPS, and CPU.

    Returns
    -------
    torch.device
    """
    if device == "auto":
        # check if MPS is available
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        # If not MPS, check if CUDA is available
        elif hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    return torch.device(device)


def count_parameters(models_or_parameters, unit="m"):
    """
    This function is to obtain the storage size unit of a (or multiple) models.

    Parameters
    ----------
    models_or_parameters : PyTorch model(s) or a list of parameters.
    unit : the storage size unit.

    Returns
    -------
    The number of parameters of the given model(s) or parameters.
    """
    if isinstance(models_or_parameters, nn.Module):
        counts = sum(v.numel() for v in models_or_parameters.parameters())
    elif isinstance(models_or_parameters, nn.Parameter):
        counts = models_or_parameters.numel()
    elif isinstance(models_or_parameters, (list, tuple)):
        return sum(count_parameters(x, unit) for x in models_or_parameters)
    else:
        counts = sum(v.numel() for v in models_or_parameters)
    unit = unit.lower()
    if unit in ("kb", "k"):
        counts /= 2**10
    elif unit in ("mb", "m"):
        counts /= 2**20
    elif unit in ("gb", "g"):
        counts /= 2**30
    elif unit is not None:
        raise ValueError("Unknown unit: {:}".format(unit))
    return counts
