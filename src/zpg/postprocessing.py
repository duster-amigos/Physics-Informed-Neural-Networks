import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from error import l2norm_err
from train_configs import ZPG_config
from loguru import logger

def postprocess() -> None:
    """
    Postprocess prediction data for ZPG PINN: plots, error calculation, and logging.
    """
    cmp = sns.color_palette('RdBu_r', as_cmap=True)
    plt.set_cmap(cmp)
    act = ZPG_config.act
    nn = ZPG_config.n_neural
    nl = ZPG_config.n_layer
    n_adam = ZPG_config.n_adam
    data_name = f"ZPG_{nn}_{nl}_{act}_{n_adam}_2400"
    data = np.load(f"pred/res_{data_name}.npz")
    les = loadmat('data/ZPG.mat')
    Retheta = les['Retheta']
    ind = np.isnan(Retheta)
    Retheta = Retheta[~ind]
    i_trn = (Retheta > 1000) & (Retheta < 7000)
    i_trn = np.squeeze(i_trn)
    id_yn = 200
    ind = ind.reshape(-1)
    u = les['U'][:id_yn, ~ind][:, i_trn]
    x = les['x'][:id_yn, ~ind][:, i_trn]
    y = les['y'][:id_yn, ~ind][:, i_trn]
    plt.figure(0)
    plt.semilogy(data["hist"][:, 0], label="Total")
    plt.semilogy(data["hist"][:, 1], label="Bc")
    plt.semilogy(data["hist"][:, 2], label="Residual")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"figs/{data_name}_Loss")
    pred = data["pred"]
    ref = data["ref"]
    names = ["U", "V", "uv"]
    for i, name in enumerate(names):
        fig, axs = plt.subplots(3, figsize=(9, 10))
        axs[0].contourf(x, y, pred[i, :, :])
        axs[1].contourf(x, y, ref[i, :, :])
        clb = axs[2].contourf(x, y, np.abs(ref[i, :, :] - pred[i, :, :]))
        cbar = plt.colorbar(clb, orientation="horizontal", pad=0.3)
        cbar.ax.locator_params(nbins=5)
        cbar.ax.tick_params(labelsize=12)
        axs[0].set_title(f"{name} Prediction", fontdict={"size": 18})
        axs[1].set_title("Reference", fontdict={"size": 18})
        axs[2].set_title("Error", fontdict={"size": 18})
        plt.tight_layout()
        plt.savefig(f"figs/{data_name}_{name}", dpi=300)
        logger.info(f"Saved contour plots for {name}.")
    err = l2norm_err(ref, pred)
    logger.info(f"Error U = {np.round(err[0], 3)}%\nError V = {np.round(err[1], 3)}%\nError uv = {np.round(err[2], 3)}%")

if __name__ == "__main__":
    postprocess()
