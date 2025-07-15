import numpy as np
from loguru import logger

def l2norm_err(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Calculate the relative L2-norm error between reference and predicted values for each variable.
    Args:
        ref (np.ndarray): Reference values, shape [N, I, ...].
        pred (np.ndarray): Predicted values, same shape as ref.
    Returns:
        np.ndarray: Relative L2-norm errors (percentage) for each variable.
    """
    error_norm = np.linalg.norm(ref - pred, axis=(1, 2))
    ref_norm = np.linalg.norm(ref, axis=(1, 2))
    relative_error = (error_norm / ref_norm) * 100
    logger.info(f"L2-norm error computed: {relative_error}")
    return relative_error
