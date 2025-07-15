from loguru import logger

class ZPG_config:
    """
    Configuration for training the ZPG Boundary Layer (BL) model.
    Attributes:
        act (str): Activation function for the MLP.
        n_adam (int): Number of Adam optimizer steps.
        n_neural (int): Number of neurons per layer.
        n_layer (int): Number of layers in the MLP.
        cp_step (int): Interval for collection points.
        method (str): Optimization method.
    """
    act: str = "tanh"
    n_adam: int = 1000
    n_neural: int = 20
    n_layer: int = 8
    cp_step: int = 500
    method: str = "L-BFGS-B"
    logger.info("ZPG_config loaded.")
