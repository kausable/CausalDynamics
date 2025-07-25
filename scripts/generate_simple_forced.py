import sys

sys.path.append("src")

from pathlib import Path
import xarray as xr
import torch
from jsonargparse import ArgumentParser

from causaldynamics.creator import (
    create_scm,
    propagate_mlp,
    logger,
)
from causaldynamics.data_io import (
    create_output_dataset,
    save_xr_dataset,
    create_dynsys_dataset,
)
from causaldynamics.utils import process_confounders, check_confounders
from causaldynamics.systems import (
    solve_random_systems,
    solve_system,
    drive_linear,
    drive_sin,
)
from causaldynamics.systems import get_adjacency_matrix_from_jac

# Edge features
_init_ratios = {
    "nonlinear": [1.0, 0.0],
    "periodic": [1.0, 1.0],
}


def generate(
    *,
    data_dir: str,
    noise: float,
    confounder: bool,
    standardize: bool,
    system_name: str,
    seed: int,
):
    """
    Generate all data, ranging from coupled models.

    This function serves as the main entry point to generate dataset, including ground truth graphs.

    Parameters
    ----------
    data_dir : str
        Directory containing the time series data and adjacency matrices in a netCDF file.

    noise : float
        Amplitude of Langevin noise; >0 is an SDE.

    confounder : bool
        Whether to apply unobserved confounding post-processing.

    standardize : bool
        Whether to standardize the trajectories following the iSCM algorithm.

    system_name : str
        The system representing each node e.g., Lorenz84 or random.

    seed : int
        Index representing SCM that is randomly constructed.

    Returns
    -------
    None
        Results are saved to disk at data_dir/output/coupled/<experiment_id>/<system_name>.nc as NetCDF files

    Notes
    -----
    Example usage:
        python scripts/generate_simple_forced.py \
                                            --data_dir data \
                                            --noise 0.5 \
                                            --confounder \
                                            --standardize \
                                            --system_name random \
                                            --seed 0

    The function saves timeseries and adjacency matrix, also other relevant metadata.
    """
    # Global hyperparameters
    num_timesteps = 1000
    num_trajectories = 10
    num_nodes = 3

    # Directory setup
    experiment_dir = (
        Path(data_dir) / "simpleforced"
        f"_noise={noise:.2f}"
        f"_systems={num_nodes}"
        f"_confounder={confounder}"
        f"_standardize={standardize}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Build the SCM
    A = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    A, W, b, root_nodes, magnitudes = create_scm(
        num_nodes=num_nodes,
        adjacency_matrix=A,
        node_dim=num_nodes,
        confounders=confounder,
    )

    # Simulate
    runs_noforcing = []
    runs_sin = []
    runs_linear = []
    for _ in range(num_trajectories):

        if system_name == "random":
            d_sys, system_names = solve_random_systems(
                num_timesteps=num_timesteps,
                num_systems=num_nodes,
                make_trajectory_kwargs={"resample": True, "noise": noise},
                return_system_names=True,
            )
        else:
            d_sys = solve_system(
                num_timesteps=num_timesteps,
                num_systems=num_nodes,
                system_name=system_name,
                make_trajectory_kwargs={"resample": True, "noise": noise},
            )
        # init = solve_random_systems(num_timesteps, num_nodes)
        if d_sys is None:
            logger.warning(f"Skipping {system_name}.")
            continue

        A_system = get_adjacency_matrix_from_jac(system_names[0])
        if A_system.ndim != 2:
            logger.warning(f"Skipping {system_name}: got ndim={A_system.ndim}")
            continue

        linear_driver = drive_linear(
            num_timesteps=num_timesteps,
            num_nodes=num_nodes,
            node_dim=3,
            m_range=(-10.0, 10.0),
            b_range=(-10.0, 10.0),
        )
        sin_driver = drive_sin(
            num_timesteps=num_timesteps,
            num_nodes=num_nodes,
            node_dim=3,
            amplitude_range=(-10.0, 10.0),
        )
        init_linear = d_sys.clone()
        init_sin = d_sys.clone()
        init_linear[:, 1] = linear_driver[:, 1]
        init_sin[:, 1] = sin_driver[:, 1]

        x_sin = propagate_mlp(A, W, b, init=init_sin, standardize=True)
        x_linear = propagate_mlp(A, W, b, init=init_linear, standardize=True)

        data_sin = xr.DataArray(x_sin, dims=["time", "node", "dim"])
        data_linear = xr.DataArray(x_linear, dims=["time", "node", "dim"])
        data_noforcing = xr.DataArray(
            d_sys[:, 0:1], dims=["time", "node", "dim"]
        )

        runs_sin.append(
            create_output_dataset(
                adjacency_matrix=A,
                weights=W,
                biases=b,
                magnitudes=magnitudes,
                time_series=data_sin,
                root_nodes=root_nodes,
                verbose=False,
            )
        )
        runs_linear.append(
            create_output_dataset(
                adjacency_matrix=A,
                weights=W,
                biases=b,
                magnitudes=magnitudes,
                time_series=data_linear,
                root_nodes=root_nodes,
                verbose=False,
            )
        )
        runs_noforcing.append(
            create_dynsys_dataset(
                adjacency_matrix=A_system,
                time_series=data_noforcing,
                verbose=False,
            )
        )

    # Post-processing: data ordering and construction
    dataset_sin = xr.concat(runs_sin, dim="system", data_vars=["time_series"])
    dataset_sin["time_series"] = dataset_sin["time_series"].transpose(
        "time", "system", "node", "dim"
    )

    dataset_linear = xr.concat(
        runs_linear, dim="system", data_vars=["time_series"]
    )
    dataset_linear["time_series"] = dataset_linear["time_series"].transpose(
        "time", "system", "node", "dim"
    )

    dataset_noforcing = xr.concat(
        runs_noforcing,
        dim="system",
        data_vars=["time_series", "adjacency_matrix"],
    )
    dataset_noforcing["time_series"] = dataset_noforcing[
        "time_series"
    ].transpose("time", "system", "dim")

    # Post-processing if confounder==True
    if confounder:
        dataset_sin = process_confounders(dataset_sin)
        dataset_linear = process_confounders(dataset_linear)
    if confounder and check_confounders(dataset_noforcing):
        dataset_noforcing = process_confounders(dataset_noforcing)

    # Save
    data_path_sin = (
        experiment_dir
        / f"data/S{system_name.upper()}_sinforcing_N{num_trajectories}_T{num_timesteps}_seed{seed}_noise{noise}.nc"
    )
    data_path_linear = (
        experiment_dir
        / f"data/S{system_name.upper()}_linearforcing_N{num_trajectories}_T{num_timesteps}_seed{seed}_noise{noise}.nc"
    )
    data_path_noforcing = (
        experiment_dir
        / f"data/S{system_name.upper()}_noforcing_N{num_trajectories}_T{num_timesteps}_seed{seed}_noise{noise}.nc"
    )
    save_xr_dataset(dataset_sin, data_path_sin)
    save_xr_dataset(dataset_linear, data_path_linear)
    save_xr_dataset(dataset_noforcing, data_path_noforcing)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--standardize",
        required=True,
        action="store_true",
        help="If passed, standardize the trajectories following the iSCM algorithm",
    )
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument(
        "--confounder",
        required=True,
        action="store_true",
        help="If passed, drop the identified confounder",
    )
    parser.add_argument("--system_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    generate(**vars(args))
