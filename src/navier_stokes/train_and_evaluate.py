import deepxde as dde
import random
import shutil
import os
import matplotlib.pyplot as plt
from loguru import logger

def random_search_and_train(data) -> None:
    """
    Perform random search for hyperparameters and train models for Navier-Stokes PINN.
    Args:
        data: PDE data for training.
    """
    layer_sizes = [
        [2] + [64] * 3 + [3],
        [2] + [128] * 3 + [3],
        [2] + [64] * 4 + [3],
        [2] + [128] * 4 + [3],
        [2] + [64] * 5 + [3],
        [2] + [128] * 5 + [3]
    ]
    activations = ["tanh", "relu"]
    initializers = ["Glorot uniform", "He normal"]
    optimizers = ["adam", "sgd"]
    learning_rates = [1e-3, 1e-4]
    num_trials = 40
    for _ in range(num_trials):
        layer_size = random.choice(layer_sizes)
        activation = random.choice(activations)
        initializer = random.choice(initializers)
        optimizer = random.choice(optimizers)
        lr = random.choice(learning_rates)
        net = dde.maps.FNN(layer_size, activation, initializer)
        model = dde.Model(data, net)
        model.compile(optimizer, lr=lr)
        losshistory, train_state = model.train(epochs=5000, display_every=10)
        config_name = f"Layers_{layer_size}_Activation_{activation}_Init_{initializer}_Optimizer_{optimizer}_LR_{lr}"
        model_save_path = os.path.join("Navier Stokes Models", f"{config_name}.h5")
        model.save(model_save_path)
        plot_save_path = os.path.join("Navier Stokes Plots", f"{config_name}_loss_plot.png")
        dde.utils.external.plot_loss_history(losshistory, fname=plot_save_path)
        logger.info(f"Model trained and saved with configuration: {config_name}")
