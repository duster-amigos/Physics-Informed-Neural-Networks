import torch
from loguru import logger
from typing import Any

def get_optimizer(optimizer_name: str, model_parameters: Any, learning_rate: float) -> torch.optim.Optimizer:
    """
    Select an optimizer by name.
    Args:
        optimizer_name (str): Name of the optimizer.
        model_parameters: Model parameters for optimization.
        learning_rate (float): Learning rate.
    Returns:
        torch.optim.Optimizer: Selected optimizer.
    """
    optimizers = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'RMSprop': torch.optim.RMSprop,
        'Adagrad': torch.optim.Adagrad,
        'AdamW': torch.optim.AdamW
    }
    logger.info(f"Using optimizer: {optimizer_name} with learning rate: {learning_rate}")
    return optimizers.get(optimizer_name, torch.optim.Adam)(model_parameters, lr=learning_rate)
