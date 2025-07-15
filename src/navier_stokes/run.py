from setup_environment import setup
from geometry_and_boundary_conditions import geom, bc_wall_u, bc_wall_v, bc_inlet_u, bc_inlet_v, bc_outlet_p, bc_outlet_v
from pde_and_data import create_data
from train_and_evaluate import random_search_and_train
from loguru import logger

def main() -> None:
    """
    Main function to execute the full PINN pipeline for Navier-Stokes.
    """
    setup()
    data = create_data(geom)
    random_search_and_train(data)
    logger.info("Navier-Stokes PINN pipeline completed.")

if __name__ == "__main__":
    main()
