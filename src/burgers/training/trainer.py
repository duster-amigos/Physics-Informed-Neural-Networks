import torch
from model.residual import pde_residual
from loguru import logger
from torch.nn import Module
from torch.optim import Optimizer

def train_model(model: Module, optimizer: Optimizer, x_train: torch.Tensor, t_train: torch.Tensor, num_epochs: int) -> Module:
    """
    Train the PINN model.
    Args:
        model (nn.Module): PINN model.
        optimizer (torch.optim.Optimizer): Optimizer.
        x_train (torch.Tensor): Training spatial points.
        t_train (torch.Tensor): Training temporal points.
        num_epochs (int): Number of training epochs.
    Returns:
        nn.Module: Trained model.
    """
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        residual = pde_residual(x_train, t_train, model)
        loss = torch.mean(residual ** 2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")
    return model
