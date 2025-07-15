import torch
from loguru import logger

def pde_residual(x: torch.Tensor, t: torch.Tensor, model: torch.nn.Module, nu: float = 0.01) -> torch.Tensor:
    """
    Compute the PDE residual for the 1D Burgers' equation:

        \[
        u_t + u u_x - \nu u_{xx} = 0
        \]

    Args:
        x (torch.Tensor): Spatial points.
        t (torch.Tensor): Temporal points.
        model (nn.Module): PINN model.
        nu (float): Viscosity parameter.

    Returns:
        torch.Tensor: Residual values.
    """
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    residual = u_t + u * u_x - nu * u_xx
    logger.debug(f"Residual computed with shape: {residual.shape}")
    return residual
