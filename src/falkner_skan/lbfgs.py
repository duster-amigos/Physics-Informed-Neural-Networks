import tensorflow as tf
import scipy.optimize as sopt
import numpy as np
from train_configs import FS_config
from loguru import logger
from typing import List, Any

class optimizer:
    """
    Optimizer for minimizing a function using a specified method (default: BFGS).
    Handles partitioning and stitching of trainable variables during optimization.
    Args:
        trainable_vars (list): List of trainable variables to optimize.
        method (str): Optimization method (default from FS_config).
    """
    def __init__(self, trainable_vars: List[Any], method: str = FS_config.method):
        super().__init__()
        self.trainable_variables = trainable_vars
        self.method = method
        self.shapes = tf.shape_n(self.trainable_variables)
        self.num_tensors = len(self.shapes)
        count = 0
        stitch_indices = []
        partition_indices = []
        for i, shape in enumerate(self.shapes):
            num_elements = np.product(shape)
            stitch_indices.append(tf.reshape(tf.range(count, count + num_elements, dtype=tf.int32), shape))
            partition_indices.extend([i] * num_elements)
            count += num_elements
        self.partition_indices = tf.constant(partition_indices)
        self.stitch_indices = stitch_indices
        logger.info(f"LBFGS optimizer initialized with {self.num_tensors} tensors.")

    def assign_params(self, params_1d: tf.Tensor) -> None:
        params_1d = tf.cast(params_1d, dtype=tf.float32)
        partitioned_params = tf.dynamic_partition(params_1d, self.partition_indices, self.num_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, partitioned_params)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))
        logger.debug("Parameters assigned to trainable variables.")

    def minimize(self, func) -> Any:
        initial_params = tf.dynamic_stitch(self.stitch_indices, self.trainable_variables)
        logger.info(f"Starting minimization using {self.method}.")
        results = sopt.minimize(
            fun=func,
            x0=initial_params,
            method=self.method,
            jac=True,
            options={
                'iprint': 0,
                'maxiter': 50000,
                'maxfun': 50000,
                'maxcor': 50,
                'maxls': 50,
                'gtol': 1.0 * np.finfo(float).eps,
                'ftol': 1.0 * np.finfo(float).eps
            }
        )
        logger.info(f"Minimization complete. Success: {results.success}")
        return results
