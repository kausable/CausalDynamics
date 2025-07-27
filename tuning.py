import sys
sys.path.append("src")

import itertools
import copy

import numpy as np
import xarray as xr
from tqdm import tqdm

from causaldynamics.baselines import (
    DYNOTEARS,
    FPCMCI,
    NGC_LSTM,
    TSCI,
    CUTSPlus,
    PCMCIPlus,
    VARLiNGAM,
    RCD,
    GIN,
    GRASP, 
    TCDF
)
from causaldynamics.creator import logger
from causaldynamics.score import score

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from jsonargparse import ArgumentParser
       

param_grids = {
    "PCMCIPlus": {
        "model_cls": PCMCIPlus,
        "grid": {
            "pc_alpha": [0.01, 0.05, 0.1],
        }
    },
    "FPCMCI": {
        "model_cls": FPCMCI,
        "grid": {
            "pc_alpha": [0.01, 0.05, 0.1],
            "f_alpha":  [0.01, 0.05, 0.1],
        }
    },
    "RCD": {
        "model_cls": RCD,
        "grid": {
            "cor_alpha": [0.01, 0.05, 0.1],
            "ind_alpha":  [0.01, 0.05, 0.1],
            "shapiro_alpha":  [0.01, 0.05, 0.1],
        }
    },
    "TSCI": {
        "model_cls": TSCI,
        "grid": {
            "embed_dim": [2, 3, 4],
            "corr_thres":  [0.7, 0.8, 0.9],
        }
    },
    "CUTSPlus": {
        "model_cls": CUTSPlus,
        "grid": {
            "corr_thres":  [0.7, 0.8, 0.9],
        }
    },
    "DYNOTEARS": {
        "model_cls": DYNOTEARS,
        "grid": {
            "lambda_w":  [0.1, 0.2, 0.3],
            "lambda_a":  [0.1, 0.2, 0.3],
        }
    },
    "NGC_LSTM": {
        "model_cls": NGC_LSTM,
        "grid": {
            "lam_ridge":  [1e-2, 1e-3, 1e-4],
            "lr":  [1e-2, 1e-3, 1e-4],
        }
    },
    "TCDF": {
        "model_cls": TCDF,
        "grid": {
            "epochs": [100, 200],
            "hidden_layers": [0, 1, 2],
            "kernel_size": [4, 5, 6],
        }
    }
}

def tuning(*, filename: str):
    """
    Tune causal discovery methods on time series data.

    This function loads time series data and ground truth adjacency matrices,
    applies a specified causal discovery method, and computes performance metrics for fine-tuning.

    Parameters
    ----------
    filename : str
        Filename containing the time series data and adjacency matrices in a netCDF file,
    causal_model : str
        Name of the causal discovery method to use

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the specified causal model is not supported

    Notes
    -----
    Example usage:
        `python tuning.py --filename data/simple/noise=0.00_confounder=False/data/Lorenz84_N10_T1000.nc`

    The function search optimal hyperparameter given validation dataset.
    """
    # Load data
    try:
        ds = xr.open_dataset(filename)
        timeseries = ds['time_series'].to_numpy().transpose(1, 0, 2) # shape of (N, T, D)
        adj_matrix = ds['adjacency_matrix'].to_numpy()
    except:
        logger.info(f"Fails to load trajectory. Either file doesnt exist or the data has incorrect format.")

    # Grid search
    best = {}
    for name, info in param_grids.items():
        cls  = info["model_cls"]
        grid = info["grid"]
        best[name] = {"joint_auroc": -np.inf, "params": None}
    
        keys, values = zip(*grid.items())
        for vals in itertools.product(*values):
            
            params = dict(zip(keys, vals))
            model = cls(**params)
            preds = []
            
            for X in tqdm(timeseries, desc=f"{name} {params}"):
                model.run(X)
                preds.append(copy.deepcopy(model.adj_matrix))
    
            df = score(np.array(preds), adj_matrix, name=name)
            ja = df.loc["Joint AUROC", name]
    
            if ja > best[name]["joint_auroc"]:
                best[name]["joint_auroc"] = ja
                best[name]["params"] = params
    
        logger.info(f"{name} best Joint AUROC = {best[name]['joint_auroc']:.4f} with params = {best[name]['params']}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--filename",
        help="Filename containing the time series data and adjacency matrices in a netCDF file",
    )
    args = parser.parse_args()
    tuning(**vars(args))

    