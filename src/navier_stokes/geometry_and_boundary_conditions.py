import numpy as np
import deepxde as dde
from loguru import logger

rho: float = 1
mu: float = 1
u_in: float = 1
D: float = 1
L: float = 2

def boundary_wall(X: np.ndarray, on_boundary: bool) -> bool:
    """
    Wall boundary condition (no-slip).
    Args:
        X (np.ndarray): Coordinates.
        on_boundary (bool): If point is on boundary.
    Returns:
        bool: True if on wall boundary.
    """
    on_wall = np.logical_and(np.logical_or(np.isclose(X[1], -D/2), np.isclose(X[1], D/2)), on_boundary)
    logger.debug(f"Wall boundary: {on_wall}")
    return on_wall

def boundary_inlet(X: np.ndarray, on_boundary: bool) -> bool:
    """
    Inlet boundary condition (specified velocity).
    """
    result = on_boundary and np.isclose(X[0], -L/2)
    logger.debug(f"Inlet boundary: {result}")
    return result

def boundary_outlet(X: np.ndarray, on_boundary: bool) -> bool:
    """
    Outlet boundary condition (zero pressure).
    """
    result = on_boundary and np.isclose(X[0], L/2)
    logger.debug(f"Outlet boundary: {result}")
    return result

geom = dde.geometry.Rectangle(xmin=[-L/2, -D/2], xmax=[L/2, D/2])
bc_wall_u = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component=0)
bc_wall_v = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component=1)
bc_inlet_u = dde.DirichletBC(geom, lambda X: u_in, boundary_inlet, component=0)
bc_inlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_inlet, component=1)
bc_outlet_p = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component=2)
bc_outlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component=1)
logger.info("Geometry and boundary conditions initialized.")
