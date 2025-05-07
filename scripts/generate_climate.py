import sys

sys.path.append("../src")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from qgs.diagnostics.multi import MultiDiagnostic
from qgs.diagnostics.streamfunctions import (
    MiddleAtmosphericStreamfunctionDiagnostic,
    OceanicLayerStreamfunctionDiagnostic,
)
from qgs.diagnostics.temperatures import (
    MiddleAtmosphericTemperatureDiagnostic,
    OceanicLayerTemperatureAnomalyDiagnostic,
)
from qgs.functions.tendencies import create_tendencies
from qgs.integrators.integrator import RungeKuttaIntegrator
from qgs.params.params import QgParams

from causaldynamics.creator import logger
from causaldynamics.data_io import save_xr_dataset
from causaldynamics.idealized.xro import XRO
from causaldynamics.utils import set_rng_seed

warnings.filterwarnings("ignore")

from jsonargparse import ArgumentParser


def generate(*, data_dir: str):
    """
    Generate all data, ranging from climate models:
        [1] Coupled atmosphere-ocean from https://github.com/Climdyn/qgs/blob/master/notebooks/maooam_run.ipynb
        [2] Coupled modes for ENSO from https://github.com/senclimate/XRO

    This function serves as the main entry point to generate dataset, including ground truth graphs.

    Parameters
    ----------
    data_dir : str
        Directory containing the time series data and adjacency matrices in a netCDF file,

    Returns
    -------
    None
        Results are saved to disk at `data_dir/output/climate/<experiment_id>/<system_name>.nc` as NetCDF files

    Notes
    -----
    Example usage:
        `python scripts/generate_climate.py --data_dir data`

    The function saves timeseries and adjacency matrix, also other relevant metadata.
    """
    logger.info("Starting the data generating process...")

    # Setting up
    DATA_DIR = Path(data_dir) / "climate"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Global hyperparameters
    num_timesteps = 1000
    num_trajectories = 10
    set_rng_seed(42)

    # 1) Run climate models (coupled atmosphere-ocean)
    logger.info("Processing coupled atmosphere-ocean systems...")
    experiment_dir = DATA_DIR / f"coupled_atmos_ocean"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    ## Case-specific hyperparameters
    dt = 0.1
    write_steps = 100

    dataset = []
    for _ in range(num_trajectories):

        ## Model setup
        model_parameters = QgParams()
        model_parameters.set_atmospheric_channel_fourier_modes(2, 2)
        model_parameters.set_oceanic_basin_fourier_modes(2, 4)
        model_parameters.set_params(
            {
                "kd": 0.0290,
                "kdp": 0.0290,
                "n": 1.5,
                "r": 1.0e-7,
                "h": 136.5,
                "d": 1.1e-7,
            }
        )
        model_parameters.atemperature_params.set_params(
            {"eps": 0.7, "T0": 289.3, "hlambda": 15.06}
        )
        model_parameters.gotemperature_params.set_params({"gamma": 5.6e8, "T0": 301.46})
        model_parameters.atemperature_params.set_insolation(103.3333, 0)
        model_parameters.gotemperature_params.set_insolation(310, 0)

        ## Integrator
        f, Df = create_tendencies(model_parameters)
        integrator = RungeKuttaIntegrator()
        integrator.set_func(f)

        ## Initial conditions (+ warmup)
        ic = np.random.rand(model_parameters.ndim) * 0.01
        ic[29] = 3.0  # Setting reasonable initial reference temperature
        ic[10] = 1.5
        integrator.integrate(0.0, 2000000.1, dt, ic=ic, write_steps=0)
        time, ic = integrator.get_trajectories()

        ## Simulate
        integrator.integrate(0.0, 500000.0, dt, ic=ic, write_steps=write_steps)
        reference_time, reference_traj = integrator.get_trajectories()
        inds = np.linspace(0, len(reference_time) - 1, num_timesteps, dtype=int)
        time = reference_time[inds]
        traj = reference_traj[:, inds]

        ## Extract states
        psi_a = MiddleAtmosphericStreamfunctionDiagnostic(
            model_parameters, delta_x=0.2, delta_y=0.2, geopotential=True
        )
        theta_a = MiddleAtmosphericTemperatureDiagnostic(
            model_parameters, delta_x=0.2, delta_y=0.2
        )
        psi_o = OceanicLayerStreamfunctionDiagnostic(
            model_parameters, delta_x=0.2, delta_y=0.2
        )
        delta_o = OceanicLayerTemperatureAnomalyDiagnostic(
            model_parameters, delta_x=0.2, delta_y=0.2
        )

        m = MultiDiagnostic(2, 3)
        m.add_diagnostic(psi_a, diagnostic_kwargs={"show_time": False})
        m.add_diagnostic(theta_a, diagnostic_kwargs={"show_time": False})
        m.add_diagnostic(psi_o, diagnostic_kwargs={"show_time": False})
        m.add_diagnostic(delta_o, diagnostic_kwargs={"show_time": False})
        m.set_data(time, traj)

        ## Build adjacency_matrix of shape (node_in, node_out)
        adjacency_matrix = np.array(
            [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1]], dtype=np.float32
        )

        ## Construct dataset
        data = np.array(
            [psi_a.diagnostic, theta_a.diagnostic, psi_o.diagnostic, delta_o.diagnostic]
        )
        data = data.astype(np.float32)
        data = data.transpose(1, 0, 2, 3)
        data = data.reshape(*data.shape[:-2], -1)
        dataset.append(data)

    dataset = np.array(dataset).transpose(1, 0, 2, 3)
    n_time, n_system, n_node, n_dim = dataset.shape

    dataset = xr.Dataset(
        {
            "time_series": (("time", "system", "node", "dim"), dataset),
            "adjacency_matrix": (("node_in", "node_out"), adjacency_matrix),
        },
        coords={
            "node_in": pd.Index(np.arange(n_node), name="node_in"),
            "node_out": pd.Index(np.arange(n_node), name="node_out"),
            "node": pd.Index(np.arange(n_node), name="node"),
            "dim_in": pd.Index(np.arange(n_dim), name="dim_in"),
            "dim_out": pd.Index(np.arange(n_dim), name="dim_out"),
            "dim": pd.Index(np.arange(n_dim), name="dim"),
            "time": pd.Index(np.arange(num_timesteps), name="time"),
            "system": pd.Index(np.arange(num_trajectories), name="system"),
        },
    )

    data_path = experiment_dir / f"data/ESM_N{num_trajectories}_T{num_timesteps}.nc"
    save_xr_dataset(dataset, data_path)

    # 2) Run climate models (coupled ENSO modes)
    logger.info("Processing coupled ENSO systems...")
    experiment_dir = DATA_DIR / f"coupled_enso_modes"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    ## Case-specific hyperparameters
    train_ds = xr.open_dataset("src/causaldynamics/idealized/XRO_indices_oras5.nc").sel(
        time=slice("1979-01", "2022-12")
    )

    coupled_vars = [
        "Nino34",
        "WWV",
        "NPMM",
        "SPMM",
        "IOB",
        "IOD",
        "SIOD",
        "TNA",
        "ATL3",
        "SASD",
    ]
    decoupled_exps = {
        "NPMM": ["NPMM"],
        "SPMM": ["SPMM"],
        "IOB": ["IOB"],
        "SIOD": ["SIOD"],
        "TNA": ["TNA"],
        "ATL3": ["ATL3"],
        "SASD": ["SASD"],
        "ExPO": ["NPMM", "SPMM"],
        "IO": ["IOB", "IOD", "SIOD"],
        "AO": ["TNA", "ATL3", "SASD"],
        "NONE": ["NPMM", "SPMM", "IOB", "IOD", "SIOD", "TNA", "ATL3", "SASD"],
    }

    for decoupled_k, decoupled_v in decoupled_exps.items():

        ## Get nonlinear terms
        nonlinear_modes = ["IOD"]

        ## Get decoupling indices (except for IOD with nonlinear terms)
        decoupled_idx = [
            coupled_vars.index(v) for v in decoupled_v if v not in nonlinear_modes
        ]

        ## Compute adjacency matrix
        adjacency_matrix = np.ones((len(coupled_vars), len(coupled_vars)))
        adjacency_matrix[:, decoupled_idx] = 0  # Set child dependency to 0

        ## Fit data and simulate
        enso_model = XRO(ncycle=12, ac_order=2)
        enso_fits = enso_model.fit_matrix(
            train_ds, maskb=nonlinear_modes, maskNT=["T2", "TH"]
        )
        enso_fits["Lac"].loc[{"rankx": decoupled_idx}] = 0  ## Decoupling
        traj = enso_model.simulate(
            fit_ds=enso_fits,
            X0_ds=train_ds.isel(time=0),
            nyear=200,
            ncopy=num_trajectories,
            is_xi_stdac=False,
            seed=42,
        )

        dataset = (
            traj.isel(time=slice(0, num_timesteps)).to_array().fillna(0).to_numpy()
        )

        dataset = dataset.astype(np.float32)
        dataset = np.expand_dims(dataset, -1).transpose(1, 2, 0, 3)
        n_time, n_system, n_node, n_dim = dataset.shape

        ## Construct dataset
        dataset = xr.Dataset(
            {
                "time_series": (("time", "system", "node", "dim"), dataset),
                "adjacency_matrix": (("node_in", "node_out"), adjacency_matrix),
            },
            coords={
                "node_in": pd.Index(np.arange(n_node), name="node_in"),
                "node_out": pd.Index(np.arange(n_node), name="node_out"),
                "node": pd.Index(np.arange(n_node), name="node"),
                "dim_in": pd.Index(np.arange(n_dim), name="dim_in"),
                "dim_out": pd.Index(np.arange(n_dim), name="dim_out"),
                "dim": pd.Index(np.arange(n_dim), name="dim"),
                "time": pd.Index(np.arange(num_timesteps), name="time"),
                "system": pd.Index(np.arange(num_trajectories), name="system"),
            },
        )

        data_path = (
            experiment_dir
            / f"data/{decoupled_k}_N{num_trajectories}_T{num_timesteps}.nc"
        )
        save_xr_dataset(dataset, data_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", help="Directory to store data outputs")
    args = parser.parse_args()
    generate(**vars(args))
