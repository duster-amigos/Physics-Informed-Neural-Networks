# ------------------------------------------------------
# MAIN PIPELINE
# Executes the pipeline for solving Burgers' equation
# ------------------------------------------------------

from data.data_loader import load_reference_data
from data.geometry import create_geometry
from data.visualization import visualize_reference_solution, visualize_geometry
from model.pinn import PINN
from training.conditions import initial_condition, boundary_condition
from training.optimizers import get_optimizer
from training.trainer import train_model

# Load data
t_ref, x_ref, exact = load_reference_data("Burgers.npz")

# Visualize reference solution
visualize_reference_solution(x_ref, t_ref, exact)

# Create geometry
points = create_geometry()

# Visualize geometry
visualize_geometry(points)

# Initialize the model
model = PINN(num_hidden_layers=3, num_neurons=50)

# Select optimizer
optimizer = get_optimizer("Adam", model.parameters(), 0.001)

# Train the model
train_model(model, optimizer, points["x_collocation"], points["t_collocation"], num_epochs=1000)
