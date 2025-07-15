from loguru import logger
import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving PDEs.

    The PINN approximates the solution to a PDE by minimizing the residual of the equation:
    
        \[
        \mathcal{L}[u(x, t)] = 0
        \]
    where \( \mathcal{L} \) is the differential operator for the target PDE.

    Args:
        num_hidden_layers (int): Number of hidden layers.
        num_neurons (int): Number of neurons per hidden layer.
    """
    def __init__(self, num_hidden_layers: int, num_neurons: int) -> None:
        super().__init__()
        logger.info(f"Initializing PINN with {num_hidden_layers} hidden layers and {num_neurons} neurons per layer.")
        layers = [nn.Linear(2, num_neurons), nn.Tanh()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(num_neurons, num_neurons), nn.Tanh()]
        layers.append(nn.Linear(num_neurons, 1))
        self.hidden = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PINN.

        Args:
            x (torch.Tensor): Spatial input.
            t (torch.Tensor): Temporal input.
        Returns:
            torch.Tensor: Network output.
        """
        inputs = torch.cat([x, t], dim=1)
        output = self.hidden(inputs)
        logger.debug(f"Forward pass output shape: {output.shape}")
        return output
