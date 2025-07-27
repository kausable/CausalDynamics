import sys
sys.path.append("src")

import copy
import warnings

import numpy as np
import pandas as pd
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

warnings.filterwarnings("ignore")

from pathlib import Path
from jsonargparse import ArgumentParser

# Ablation control
ABLATION_PARAMS = {
    "sampling": [1, 5, 10],
    "observability": [lambda x: x[..., 0], lambda x: x.mean(axis=-1)]
}


def ablate(*, exp_dir: str, abl_type: str):
    """
    Running a few more ablation to check model sensitivity to time discretization and observability.

    Parameters
    ----------
    exp_dir : str
        Directory containing the ground truth.

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
        python ablate.py \
            --exp_dir data/coupled/coupling=nonlinear_noise=2.00_systems=10_confounder=True_standardize=True_timelag=1 \
            --abl_type sampling
    """
    exp_dir = Path(exp_dir)
    DATA_DIR = exp_dir / "data"
    DYN_SYSTEMS = list(DATA_DIR.glob("*.nc"))

    # your causal models
    causal_models = {
        "pcmciplus": PCMCIPlus(),
        "fpcmci":    FPCMCI(),
        "varlingam": VARLiNGAM(),
        "dynotears": DYNOTEARS(),
        "ngc_lstm":  NGC_LSTM(),
        "tsci":      TSCI(),
        "cutsplus":  CUTSPlus(),
        "rcd":       RCD(),
        "grasp":     GRASP(),
        "tcdf":      TCDF(),
    }

    # Pick the list of ablation parameters
    ablation_param = ABLATION_PARAMS.get(abl_type, None)
    if ablation_param is None:
        raise ValueError(f"Unknown abl_type {abl_type}")

    # Store results
    records = []

    for abl_p in ablation_param:
        
        if callable(abl_p):
            param_name = abl_p.__name__ if abl_p.__name__ != "<lambda>" else repr(abl_p)
        else:
            param_name = str(abl_p)

        for model_name, model in causal_models.items():
            
            all_scores = []

            for dyn_system in DYN_SYSTEMS:
                ds = xr.open_dataset(dyn_system)
                timeseries = ds["time_series"].to_numpy()

                # Observability vs sampling
                if timeseries.ndim == 4:
                    if abl_type == "observability":
                        timeseries = abl_p(timeseries)
                    else:
                        timeseries = timeseries[..., 0]

                timeseries = timeseries.transpose(1, 0, 2) # now (N, T, D)  

                if abl_type == "sampling":
                    timeseries = timeseries[:, ::abl_p]

                timeseries = np.nan_to_num(timeseries)
                timeseries = (timeseries - timeseries.mean((0,1), keepdims=True)) / (timeseries.std((0,1), keepdims=True) + 1e-8)

                # Pick adjacency summary or full
                if "adjacency_matrix_summary" in ds:
                    adj = ds["adjacency_matrix_summary"].to_numpy()
                else:
                    adj = ds["adjacency_matrix"].to_numpy()

                # Skip trivial truths
                if np.all(adj==0) or np.all(adj==1):
                    continue

                # Run inference on _each_ trajectory, build preds
                preds = []
                for x in timeseries:
                    try:
                        model.run(X=x)
                        preds.append(copy.deepcopy(model.adj_matrix))
                    except Exception:
                        logger.info(f"Model {model_name} failed on {dyn_system.name}")
                        preds.append(np.zeros_like(adj))

                # Score for _this_ dynamical system
                try:
                    df = score(np.array(preds), adj, name=model_name)
                except:
                    # if scoring fails, zeros
                    df = score(
                        preds=np.zeros((timeseries.shape[0], *adj.shape), dtype=adj.dtype),
                        labs=adj,
                        name=model_name,
                    )

                # Extract metrics
                vec = df[model_name].values  # [Joint AUROC, Ind AUROC, Null AUROC, Joint AUPRC, ..., Joint SHD]
                all_scores.append(vec)

            if not all_scores:
                continue

            # Average across systems
            all_scores = np.vstack(all_scores)  # shape (n_systems, n_models)
            mean_metrics = all_scores.mean(axis=0)

            # Record one line
            records.append({
                "abl_type": abl_type,
                "abl_param": param_name,
                "model": model_name,
                "Joint_AUROC": mean_metrics[0],
                "Individual_AUROC": mean_metrics[1],
                "Null_AUROC": mean_metrics[2],
                "Joint_AUPRC": mean_metrics[3],
                "Individual_AUPRC": mean_metrics[4],
                "Null_AUPRC": mean_metrics[5],
                "Joint_SHD": mean_metrics[6],
            })

    # Save
    df = pd.DataFrame.from_records(records)
    df.to_csv(Path("tmp") / f"ablation_summary_{abl_type}.csv", index=False)
    

if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument(
        "--exp_dir",
        required=True,
        help="Root experiment directory",
    )
    p.add_argument(
        "--abl_type",
        required=True,
        help="Ablation type, one of [sampling, observability]"
    )
    args = p.parse_args()
    ablate(**vars(args))
