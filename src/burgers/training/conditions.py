import torch
import numpy as np
from loguru import logger

def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """
    Initial condition for Burgers' equation:
        \[
        u(x, 0) = -\sin(\pi x)
        \]
    Args:
        x (torch.Tensor): Spatial points.
    Returns:
        torch.Tensor: Initial condition values.
    """
    result = -torch.sin(np.pi * x)
    logger.debug(f"Initial condition computed for shape: {x.shape}")
    return result

def boundary_condition(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Boundary condition for Burgers' equation:
        \[
        u(-1, t) = u(1, t) = 0
        \]
    Args:
        x (torch.Tensor): Spatial points.
        t (torch.Tensor): Temporal points.
    Returns:
        torch.Tensor: Boundary condition values.
    """
    result = torch.zeros_like(t)
    logger.debug(f"Boundary condition computed for shape: {t.shape}")
    return result
