import sys
sys.path.append("src")


import numpy as np
import pandas as pd
import xarray as xr
from causaldynamics.creator import logger
from causaldynamics.score import score

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from jsonargparse import ArgumentParser


def diagnose(*, exp_dir: str):
    """
    Diagnose causal discovery methods on specific experimental setup.

    This function loads ground truth and its corresponding output, and summarizes the scores.

    Parameters
    ----------
    exp_dir : str
        Directory containing the ground truth and its corresponding prediction.

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
        python diagnose.py --exp_dir data/simple/noise=0.00_confounder=False
    """
    exp_dir = Path(exp_dir)
    eval_dir = exp_dir / "eval"

    # Summary statistics for each model given an experiment identifier
    records = []
    for model_dir in sorted(eval_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for nc in sorted(model_dir.glob("*.nc")):
            ds = xr.open_dataset(nc)
            metrics = {var: float(ds[var].item()) for var in ds.data_vars}
            metrics["model"]   = model
            metrics["dataset"] = nc.stem
            records.append(metrics)

    df = pd.DataFrame.from_records(records)
    df = df.set_index(["model", "dataset"])
    df = df.groupby(level="model").mean()
    print(df.to_string(float_format="%.3f"))
    

if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument(
        "--exp_dir",
        required=True,
        help="Root experiment directory",
    )
    args = p.parse_args()
    diagnose(**vars(args))
