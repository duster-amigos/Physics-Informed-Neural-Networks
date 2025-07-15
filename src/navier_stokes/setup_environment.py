import os
import shutil
from loguru import logger

def install_libraries() -> None:
    """
    Install required libraries for DeepXDE and TensorFlow.
    """
    os.system('pip install deepxde tensorflow')
    logger.info("Required libraries installed.")

def set_deepxde_backend() -> None:
    """
    Set DeepXDE backend to TensorFlow.
    """
    from deepxde.backend.set_default_backend import set_default_backend
    set_default_backend("tensorflow")
    logger.info("DeepXDE backend set to TensorFlow.")

def delete_folder(folder_path: str) -> None:
    """
    Delete a folder and its contents if it exists.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        logger.info(f"Deleted folder: {folder_path}")

def create_folders() -> None:
    """
    Create directories for saving models and plots if they don't exist.
    """
    os.makedirs("Navier Stokes Models", exist_ok=True)
    os.makedirs("Navier Stokes Plots", exist_ok=True)
    logger.info("Created folders for models and plots.")

def setup() -> None:
    install_libraries()
    set_deepxde_backend()
    create_folders()
    logger.info("Environment setup complete.")

if __name__ == "__main__":
    setup()