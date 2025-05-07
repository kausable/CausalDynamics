import sys

sys.path.append("src")

import copy
import warnings

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
)
from causaldynamics.creator import logger
from causaldynamics.score import score

warnings.filterwarnings("ignore")

from pathlib import Path

from jsonargparse import ArgumentParser

CAUSAL_MODELS = [
    "pcmciplus",
    "fpcmci",
    "varlingam",
    "dynotears",
    "ngc_lstm",
    "tsci",
    "cutsplus",
]


def evaluate(*, data_dir: str, causal_model: str):
    """
    Evaluate causal discovery methods on time series data.

    This function loads time series data and ground truth adjacency matrices,
    applies a specified causal discovery method, and computes performance metrics.

    Parameters
    ----------
    data_dir : str
        Directory containing the time series data and adjacency matrices in a netCDF file,
    causal_model : str
        Name of the causal discovery method to use

    Returns
    -------
    None
        Results are saved to disk at `data_dir/eval/{causal_model}/<system_name>.nc` as NetCDF files

    Raises
    ------
    ValueError
        If the specified causal model is not supported

    Notes
    -----
    Example usage:
        `python eval.py --data_dir data/climate/coupled_atmos_ocean --causal_model pcmciplus`

    The function saves evaluation metrics including AUROC and AUPRC scores for each system.
    """
    # Check if causal model is supported
    if causal_model not in CAUSAL_MODELS:
        raise ValueError(
            f"Causal model {causal_model} not supported. Supported models: {CAUSAL_MODELS}"
        )

    # Setting up
    DATA_DIR = Path(data_dir) / "data"
    EVAL_DIR = Path(data_dir) / "eval" / causal_model
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    DYN_SYSTEMS = list(DATA_DIR.glob(f"*.nc"))

    # Initialize causal model
    causal_models = {
        "pcmciplus": PCMCIPlus(tau_max=1, pc_alpha=0.05),
        "fpcmci": FPCMCI(tau_max=1, pc_alpha=0.05),
        "varlingam": VARLiNGAM(tau_max=1),
        "dynotears": DYNOTEARS(tau_max=1),
        "ngc_lstm": NGC_LSTM(tau_max=1),
        "tsci": TSCI(tau_max=1, corr_thres=0.8),
        "cutsplus": CUTSPlus(tau_max=1, corr_thres=0.8),
    }

    # Run summary graph inference
    for dyn_system in DYN_SYSTEMS:

        ## Load data (timeseries and adjacency matrix)
        input_ds = xr.open_dataset(dyn_system)
        timeseries = input_ds["time_series"].to_numpy()

        ## Subsample one variable if multi-dimensional (ie., partially observed / unresolved systems)
        if timeseries.ndim == 4:
            timeseries = timeseries[..., 0]
        timeseries = timeseries.transpose(1, 0, 2)  # of shape (N, T, D)

        ## Handle missing values
        timeseries = np.nan_to_num(timeseries)

        ## z-standardize for stability
        timeseries = (timeseries - timeseries.mean(axis=(0, 1), keepdims=True)) / (
            timeseries.std(axis=(0, 1), keepdims=True) + 1e-8
        )

        ## Extract adjacency matrix
        ## NOTE: skip if adjacency matrix are all ones (AUROC is unable to process singular truth value)
        adj_matrix = input_ds["adjacency_matrix"].to_numpy()
        if np.all(adj_matrix == 1) or np.all(adj_matrix == 0):
            continue

        ## Infer graph for each trajectory
        ## NOTE: safe run -- assigns zeros for trajectory-level estimated graph if run fails
        est_adj_matrix = []
        for x in tqdm(timeseries):
            model = causal_models.get(causal_model)

            try:
                model.run(X=x)
                est_adj_matrix.append(copy.deepcopy(model.adj_matrix))

            except:
                logger.info(f"Fails for a trajectory in {dyn_system}...")
                est_adj_matrix.append(np.zeros_like(adj_matrix))

        ## Compute scores
        ### NOTE: safe eval -- assigns zeros for all estimated graph if evaluation fails
        try:
            est_score = score(
                preds=np.array(est_adj_matrix), labs=adj_matrix, name=model
            )

        except:
            est_score = score(
                preds=np.zeros(
                    (timeseries.shape[0], *adj_matrix.shape), dtype=adj_matrix.dtype
                ),
                labs=adj_matrix,
                name=model,
            )

        ## Save
        est_score = est_score[[model]].values.squeeze()
        eval_ds = xr.Dataset(
            data_vars={
                "Joint_AUROC": est_score[0],
                "Individual_AUROC": est_score[1],
                "Null_AUROC": est_score[2],
                "Joint_AUPRC": est_score[3],
                "Individual_AUPRC": est_score[4],
                "Null_AUPRC": est_score[5],
            },
            attrs={"description": f"Causal discovery performance metrics ({model})"},
        )

        eval_ds.to_netcdf(EVAL_DIR / f"{dyn_system.stem}.nc")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Directory containing the time series data and adjacency matrices in a netCDF file",
    )
    parser.add_argument(
        "--causal_model",
        help="Name of the causal discovery method to use",
        choices=CAUSAL_MODELS,
    )
    args = parser.parse_args()
    evaluate(**vars(args))
