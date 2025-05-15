import sys
sys.path.append("src")

from pathlib import Path

import xarray as xr
from jsonargparse import ArgumentParser

from causaldynamics.creator import create_scm, simulate_system
from causaldynamics.data_io import create_output_dataset, save_xr_dataset
from causaldynamics.utils import process_confounders

# Edge features
_init_ratios = {
    "nonlinear": [1.0, 0.0],
    "periodic": [1.0, 1.0],
}


def generate(
    *,
    data_dir: str,
    coupling_n: str,
    noise: float,
    num_nodes: int,
    confounder: bool,
    standardize: bool,
    system_name: str,
    time_lag: int,
    seed: int,
):
    """
    Generate all data, ranging from coupled models.

    This function serves as the main entry point to generate dataset, including ground truth graphs.

    Parameters
    ----------
    data_dir : str
        Directory containing the time series data and adjacency matrices in a netCDF file.

    coupling_n : str
        The type of coupling between nodes, along edges; e.g., nonlinear, periodic.

    noise : float
        Amplitude of Langevin noise; >0 is an SDE.

    num_nodes : int
        Number of multi-dimensional nodes, each representing a system.

    confounder : bool
        Whether to apply unobserved confounding post-processing.

    standardize : bool
        Whether to standardize the trajectories following the iSCM algorithm.

    time_lag: int
        The time lag of the system, if applicable.

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
        python scripts/generate_coupled.py \
                                            --data_dir data \
                                            --coupling_n nonlinear \
                                            --noise 0.5 \
                                            --num_nodes 3 \
                                            --confounder \
                                            --time_lag 0 \
                                            --standardize \
                                            --system_name random \
                                            --seed 0

    The function saves timeseries and adjacency matrix, also other relevant metadata.
    """
    # Global hyperparameters
    num_timesteps = 1000
    num_trajectories = 10

    # Directory setup
    experiment_dir = (
        Path(data_dir)
        / "coupled"
        / f"coupling={coupling_n}_noise={noise:.2f}_systems={num_nodes}_confounder={confounder}_standardize={standardize}_timelag={time_lag}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Build the SCM
    A, W, b, root_nodes, magnitudes = create_scm(
        num_nodes=num_nodes,
        node_dim=3,
        confounders=confounder,
        graph="scale-free" if num_nodes > 3 else "all_uniform",
        time_lag=time_lag,
        time_lag_edge_probability=0.1,
    )

    # Simulate
    runs = []
    for _ in range(num_trajectories):
        da = simulate_system(
            A,
            W,
            b,
            num_timesteps=num_timesteps,
            num_nodes=num_nodes,
            init_ratios=_init_ratios[coupling_n],
            init=None,
            standardize=standardize,
            system_name=system_name,
            time_lag=time_lag,
            make_trajectory_kwargs={"resample": True, "noise": noise},
        )

        runs.append(
            create_output_dataset(
                adjacency_matrix=A,
                weights=W,
                biases=b,
                magnitudes=magnitudes,
                time_lag=time_lag,
                time_series=da,
                root_nodes=root_nodes,
                verbose=False,
            )
        )

    # Post-processing: data ordering and construction
    dataset = xr.concat(runs, dim="system", data_vars=["time_series"])
    dataset["time_series"] = dataset["time_series"].transpose(
        "time", "system", "node", "dim"
    )

    # Post-processing if confounder==True
    if confounder:
        dataset = process_confounders(dataset)

    # Save
    data_path = (
        experiment_dir
        / f"data/S{system_name.upper()}_N{num_trajectories}_T{num_timesteps}_seed{seed}.nc"
    )
    save_xr_dataset(dataset, data_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--standardize",
        required=True,
        action="store_true",
        help="If passed, standardize the trajectories following the iSCM algorithm",
    )
    parser.add_argument("--coupling_n", required=True, choices=list(_init_ratios))
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument(
        "--confounder",
        required=True,
        action="store_true",
        help="If passed, drop the identified confounder",
    )
    parser.add_argument("--time_lag", type=int, required=True)
    parser.add_argument("--system_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    generate(**vars(args))
