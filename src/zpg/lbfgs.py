import tensorflow as tf
import scipy.optimize as sopt
import numpy as np
from train_configs import ZPG_config
from loguru import logger
from typing import List, Any

class Optimizer:
    """
    Custom optimizer for variable stitching and partitioning using SciPy's optimization methods.
    Args:
        trainable_vars (list): List of trainable TensorFlow variables.
        method (str): Optimization method (default from ZPG_config).
    """
    def __init__(self, trainable_vars: List[Any], method: str = ZPG_config.method):
        super().__init__()
        self.trainable_variables = trainable_vars
        self.method = method
        self.shapes = tf.shape_n(self.trainable_variables)
        self.n_tensors = len(self.shapes)
        count = 0
        idx = []
        part = []
        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            part.extend([i] * n)
            count += n
        self.part = tf.constant(part)
        self.idx = idx
        logger.info(f"LBFGS optimizer initialized with {self.n_tensors} tensors.")

    def assign_params(self, params_1d: tf.Tensor) -> None:
        params_1d = tf.cast(params_1d, dtype=tf.float32)
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, params)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))
        logger.debug("Parameters assigned to trainable variables.")

    def minimize(self, func) -> Any:
        init_params = tf.dynamic_stitch(self.idx, self.trainable_variables)
        logger.info(f"Starting minimization using {self.method}.")
        results = sopt.minimize(
            fun=func,
            x0=init_params,
            method=self.method,
            jac=True,
            options={
                'iprint': 0,
                'maxiter': 50000,
                'maxfun': 50000,
                'maxcor': 50,
                'maxls': 50,
                'gtol': 1.0 * np.finfo(float).eps,
                'ftol': 1.0 * np.finfo(float).eps,
            }
        )
        logger.info(f"Minimization complete. Success: {results.success}")
        return results
