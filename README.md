# Physics-Informed Neural Networks (PINNs) Suite

A comprehensive, modular suite of Physics-Informed Neural Networks (PINNs) for solving classic fluid dynamics problems:
- **Burgers' Equation**
- **Falkner–Skan Boundary Layer**
- **Navier-Stokes Equations**
- **Zero-Pressure-Gradient (ZPG) Boundary Layer**

## Features
- Modular, extensible codebase for multiple PDEs
- Clean, well-documented Python code with type hints and logging
- Reproducible experiments and results
- Ready-to-use scripts for training, evaluation, and visualization
- Reference datasets and result plots

## Project Structure

```
PINNS/
│
├── src/
│   ├── burgers/                # PINN for Burgers' Equation
│   │   ├── main.py             # Main pipeline for Burgers' PINN
│   │   ├── model/              # Model architecture and PDE residual
│   │   │   ├── pinn.py         # PINN neural network definition
│   │   │   └── residual.py     # PDE residual computation
│   │   ├── training/           # Training utilities
│   │   │   ├── trainer.py      # Training loop
│   │   │   ├── optimizers.py   # Optimizer selection
│   │   │   └── conditions.py   # Initial and boundary conditions
│   │   ├── data/               # Data utilities
│   │   │   ├── data_loader.py  # Loads reference data
│   │   │   ├── geometry.py     # Generates collocation/boundary points
│   │   │   └── visualization.py# Visualization functions
│   │   └── Burgers.npz         # Reference solution data
│   │
│   ├── falkner_skan/           # PINN for Falkner–Skan boundary layer
│   │   ├── PINN_FS.py          # PINN class for Falkner–Skan
│   │   ├── train.py            # Training script
│   │   ├── train_configs.py    # Training configuration
│   │   ├── postprocessing.py   # Postprocessing and visualization
│   │   ├── lbfgs.py            # L-BFGS optimizer utility
│   │   ├── error.py            # L2-norm error calculation
│   │   └── data/               # Reference data
│   │       └── Falkner_Skan_Ref_Data.npz
│   │
│   ├── navier_stokes/          # PINN for Navier-Stokes equations
│   │   ├── geometry_and_boundary_conditions.py # Geometry and BCs
│   │   ├── pde_and_data.py     # PDE definition and data creation
│   │   ├── train_and_evaluate.py # Training and hyperparameter search
│   │   ├── setup_environment.py  # Environment setup
│   │   ├── run.py              # Main pipeline
│   │   └── Plots/              # Saved loss plots
│   │
│   └── zpg/                    # PINN for ZPG boundary layer
│       ├── PINN_ZPG.py         # PINN class for ZPG
│       ├── train.py            # Training script
│       ├── train_configs.py    # Training configuration
│       ├── postprocessing.py   # Postprocessing and visualization
│       ├── lbfgs.py            # L-BFGS optimizer utility
│       ├── error.py            # L2-norm error calculation
│       └── data/               # Reference data (see file for download link)
│
├── notebooks/                  # Jupyter notebooks for analysis
├── results/                    # Output plots and results
├── ProjectReport.pdf           # Project Report for this Project 
├── requirements.txt            # Python dependencies
├── LICENSE                     # License (personal, non-distributable)
├── README.md                   # Project documentation
```

## Setup
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd PINNS
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Each submodule contains a `main.py` or `train.py` for training and evaluation. Example:
```bash
python src/burgers/main.py
```

## Results
Please refer the project report for results


## License
This project is for personal, academic, and reference use only. Redistribution is not permitted. See LICENSE for details.
