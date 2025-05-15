import sys

sys.path.append("../src")

import warnings
from pathlib import Path

import dysts.flows as flows
import numpy as np
import xarray as xr

from causaldynamics.creator import logger
from causaldynamics.data_io import create_dynsys_dataset, save_xr_dataset
from causaldynamics.systems import (
    _DYSTS_3D_CHAOTIC_SYSTEMS,
    get_adjacency_matrix_from_jac,
    solve_system,
)
from causaldynamics.utils import check_confounders, process_confounders, set_rng_seed

warnings.filterwarnings("ignore")

from jsonargparse import ArgumentParser


def generate(*, data_dir: str, noise: float, confounder: bool, system_name: str):
    """
    Generate all data, ranging from uncoupled (simple) models.

    This function serves as the main entry point to generate dataset, including ground truth graphs.

    Parameters
    ----------
    data_dir : str
        Directory containing the time series data and adjacency matrices in a netCDF file.

    noise : float
        Amplitude of Langevin noise; >0 is an SDE.

    confounder : bool
        Whether to apply unobserved confounding post-processing.

    system_name : str
        The system under study e.g., Lorenz84.

    Returns
    -------
    None
        Results are saved to disk at data_dir/output/simple/<experiment_id>/<system_name>.nc as NetCDF files

    Notes
    -----
    Example usage:
        python scripts/generate_simple.py --data_dir data --noise 0.5 --confounder --system_name Lorenz84

    The function saves timeseries and adjacency matrix, also other relevant metadata.
    """

    # Global hyperparameters
    num_timesteps = 1000
    num_trajectories = 10
    set_rng_seed(42)

    # Directory setup
    experiment_dir = (
        Path(data_dir) / "simple" / f"noise={noise:.2f}_confounder={confounder}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Get adjacency matrix
    system = getattr(flows, system_name)
    A = get_adjacency_matrix_from_jac(system)
    if A.ndim != 2:
        logger.warning(f"Skipping {system_name}: got ndim={A.ndim}")
        return

    logger.info(
        f"Solving for {system_name}, noise={noise:.2f}, confounder={confounder}"
    )

    # Simulate
    da = solve_system(
        num_timesteps,
        num_trajectories,
        system_name,
        make_trajectory_kwargs={"resample": True, "noise": noise},
    )

    if da is None:
        return

    # Post-processing: data ordering and construction
    da = xr.DataArray(da, dims=["time", "system", "dim"])
    ds = create_dynsys_dataset(adjacency_matrix=A, time_series=da, verbose=False)

    # Post-processing if confounder==True
    if confounder and check_confounders(ds):
        ds = process_confounders(ds)

    # Save
    data_path = experiment_dir / f"data/{system_name}_N10_T1000.nc"
    save_xr_dataset(ds, data_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--confounder", required=True, action="store_true")
    parser.add_argument("--system_name", required=True)
    args = parser.parse_args()
    generate(**vars(args))
