import deepxde as dde
import numpy as np
from loguru import logger

def pde(X: np.ndarray, Y: np.ndarray) -> list:
    """
    Governing PDEs for velocity and pressure fields:
        \[
        u u_x + v u_y + \frac{1}{\rho} p_x - \frac{\mu}{\rho} (u_{xx} + u_{yy}) = 0
        \]
        \[
        u v_x + v v_y + \frac{1}{\rho} p_y - \frac{\mu}{\rho} (v_{xx} + v_{yy}) = 0
        \]
        \[
        u_x + v_y = 0
        \]
    Args:
        X (np.ndarray): Coordinates.
        Y (np.ndarray): Field values (u, v, p).
    Returns:
        list: Residuals for each equation.
    """
    du_x = dde.grad.jacobian(Y, X, i=0, j=0)
    du_y = dde.grad.jacobian(Y, X, i=0, j=1)
    dv_x = dde.grad.jacobian(Y, X, i=1, j=0)
    dv_y = dde.grad.jacobian(Y, X, i=1, j=1)
    dp_x = dde.grad.jacobian(Y, X, i=2, j=0)
    dp_y = dde.grad.jacobian(Y, X, i=2, j=1)
    du_xx = dde.grad.hessian(Y, X, i=0, j=0, component=0)
    du_yy = dde.grad.hessian(Y, X, i=1, j=1, component=0)
    dv_xx = dde.grad.hessian(Y, X, i=0, j=0, component=1)
    dv_yy = dde.grad.hessian(Y, X, i=1, j=1, component=1)
    pde_u = Y[:, 0:1] * du_x + Y[:, 1:2] * du_y + 1 / 1 * dp_x - (1 / 1) * (du_xx + du_yy)
    pde_v = Y[:, 0:1] * dv_x + Y[:, 1:2] * dv_y + 1 / 1 * dp_y - (1 / 1) * (dv_xx + dv_yy)
    pde_cont = du_x + dv_y
    logger.debug("PDE residuals computed.")
    return [pde_u, pde_v, pde_cont]

def create_data(geom) -> dde.data.PDE:
    """
    Create PDE data for training.
    Args:
        geom: Geometry object.
    Returns:
        dde.data.PDE: PDE data for DeepXDE.
    """
    from geometry_and_boundary_conditions import bc_wall_u, bc_wall_v, bc_inlet_u, bc_inlet_v, bc_outlet_p, bc_outlet_v
    data = dde.data.PDE(
        geom,
        pde,
        [bc_wall_u, bc_wall_v, bc_inlet_u, bc_inlet_v, bc_outlet_p, bc_outlet_v],
        num_domain=2000,
        num_boundary=200,
        num_test=100
    )
    logger.info("PDE data created for training.")
    return data
