import numpy as np
from loguru import logger
from typing import Tuple

def load_reference_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load reference data for the Burgers' equation.
    Args:
        file_path (str): Path to the .npz file containing reference data.
    Returns:
        tuple: t_ref (time meshgrid), x_ref (space meshgrid), exact (solution values).
    """
    data = np.load(file_path)
    t_ref, x_ref, exact = data["t"], data["x"], data["usol"].T
    x_ref, t_ref = np.meshgrid(x_ref, t_ref)
    logger.info(f"Loaded reference data from {file_path} with shapes: t_ref={t_ref.shape}, x_ref={x_ref.shape}, exact={exact.shape}")
    return t_ref, x_ref, exact
