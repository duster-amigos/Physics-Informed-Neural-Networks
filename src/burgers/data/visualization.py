import matplotlib.pyplot as plt
from loguru import logger
import torch
from typing import Dict

def visualize_reference_solution(x_ref: torch.Tensor, t_ref: torch.Tensor, exact: torch.Tensor) -> None:
    """
    Plot the reference solution as a contour plot.
    Args:
        x_ref (torch.Tensor): Space meshgrid.
        t_ref (torch.Tensor): Time meshgrid.
        exact (torch.Tensor): Solution values.
    """
    plt.figure(figsize=(16, 4))
    plt.contourf(x_ref.cpu().numpy(), t_ref.cpu().numpy(), exact.cpu().numpy(), levels=250, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title("Reference Solution of Burgers' Equation")
    plt.show()
    logger.info("Reference solution visualized.")

def visualize_geometry(points: Dict[str, torch.Tensor]) -> None:
    """
    Plot the collocation and boundary points.
    Args:
        points (dict): Dictionary containing geometry points.
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(points["x_collocation"].cpu().numpy(), points["t_collocation"].cpu().numpy(), label='Domain Points', color='blue', s=10)
    plt.scatter(points["x_initial_condition"].cpu().numpy(), points["t_initial_condition"].cpu().numpy(), label='Initial Condition Points', color='green', s=30)
    plt.scatter(points["x_boundary_left"].cpu().numpy(), points["t_boundary_points"].cpu().numpy(), label='Left Boundary Points', color='red', s=30)
    plt.scatter(points["x_boundary_right"].cpu().numpy(), points["t_boundary_points"].cpu().numpy(), label='Right Boundary Points', color='orange', s=30)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Collocation Points in the Domain and on the Boundaries')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
    logger.info("Geometry visualized.")
