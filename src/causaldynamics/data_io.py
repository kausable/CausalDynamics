from pathlib import Path

import numpy as np
import xarray as xr

from causaldynamics.scm import add_root_self_edges

def create_output_dataset(
    *,
    adjacency_matrix: np.ndarray = None,
    weights: np.ndarray = None,
    biases: np.ndarray = None,
    magnitudes: np.ndarray = None,
    time_series: np.ndarray = None,
    root_nodes: np.ndarray = None,
    time_lag: int = None,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Create an xarray Dataset containing all the components of the simulation.

    Parameters
    ----------
    adjacency_matrix : np.ndarray, optional
        The adjacency matrix representing the directed graph structure, shape (num_nodes, num_nodes).
    weights : np.ndarray, optional
        The weight matrices for each node in the MLP, shape (num_nodes, node_dim, node_dim).
    biases : np.ndarray, optional
        The bias vectors for each node in the MLP, shape (num_nodes, node_dim).
    time_series : np.ndarray, optional
        The time series data for all nodes, shape (num_timesteps, num_nodes, node_dim).
    root_nodes : np.ndarray, optional
        Boolean mask indicating which nodes are root nodes, shape (num_nodes,).
    time_lag : int, optional
        The time lag of the system, if applicable.
    magnitudes : np.ndarray, optional
        The magnitudes of the weights for each node in the MLP, shape (num_nodes,).
    Returns
    -------
    xr.Dataset
        A dataset containing all components with appropriate coordinates.
    """
    if verbose:
        print(
            "Saving adjacency matrix of shape:",
            adjacency_matrix.shape if adjacency_matrix is not None else None,
        )
        print(
            "Saving weights of shape:", weights.shape if weights is not None else None
        )
        print("Saving biases of shape:", biases.shape if biases is not None else None)
        print("Saving time lag:", time_lag)
        print(
            "Saving magnitudes of shape:",
            magnitudes.shape if magnitudes is not None else None,
        )
        print(
            "Saving time series of shape:",
            time_series.shape if time_series is not None else None,
        )
        print("Saving root nodes:", root_nodes)

    # Treat time lag of 0 as no time lag
    if time_lag is None:
        time_lag = 0

    # Create coordinates
    coords = {}
    data_vars = {}

    if adjacency_matrix is not None:
        if adjacency_matrix.ndim == 2:
            coords["node_in"] = np.arange(adjacency_matrix.shape[0])
            coords["node_out"] = np.arange(adjacency_matrix.shape[1])
            data_vars["adjacency_matrix"] = (["node_in", "node_out"], adjacency_matrix)
            data_vars["adjacency_matrix_summary"] = (["node_in", "node_out"], add_root_self_edges(adjacency_matrix))
        elif adjacency_matrix.ndim == 3:
            A = adjacency_matrix.sum(axis=0)
            # Replace any double edges with single ones in the adjacency matrix
            A = np.where(A == 2, 1, A)
            coords["node_in"] = np.arange(A.shape[0])
            coords["node_out"] = np.arange(A.shape[1])
            data_vars["adjacency_matrix"] = (["node_in", "node_out"], A)
            data_vars["adjacency_matrix_summary"] = (["node_in", "node_out"], add_root_self_edges(A))
            data_vars["adjacency_matrix_regular_edges"] = (
                ["node_in", "node_out"],
                adjacency_matrix[0],
            )
            data_vars["adjacency_matrix_time_edges"] = (
                ["node_in", "node_out"],
                adjacency_matrix[1],
            )

    if weights is not None:
        if time_lag == 0:
            coords["node"] = np.arange(weights.shape[0])
            coords["dim_in"] = np.arange(weights.shape[1])
            coords["dim_out"] = np.arange(weights.shape[2])
            data_vars["weights"] = (["node", "dim_in", "dim_out"], weights)
        else:
            coords["lag"] = np.array([False, True])
            weights = weights.reshape(
                2, weights.shape[0] // 2, weights.shape[1], weights.shape[2]
            )
            coords["node"] = np.arange(weights.shape[1])
            coords["dim_in"] = np.arange(weights.shape[2])
            coords["dim_out"] = np.arange(weights.shape[3])
            data_vars["weights"] = (["lag", "node", "dim_in", "dim_out"], weights)

    if biases is not None:
        if time_lag == 0:
            if "node" not in coords and biases.shape[0] > 0:
                coords["node"] = np.arange(biases.shape[0])
            coords["dim"] = np.arange(biases.shape[1])
            data_vars["biases"] = (["node", "dim"], biases)
        else:
            coords["lag"] = np.array([False, True])
            biases = biases.reshape(2, biases.shape[0] // 2, biases.shape[1])
            if "node" not in coords and biases.shape[0] > 0:
                coords["node"] = np.arange(biases.shape[0])
            coords["dim"] = np.arange(biases.shape[-1])
            data_vars["biases"] = (["lag", "node", "dim"], biases)

    if magnitudes is not None:
        if time_lag == 0:
            if "node" not in coords and magnitudes.shape[0] > 0:
                coords["node"] = np.arange(magnitudes.shape[0])
            data_vars["magnitudes"] = (["node"], magnitudes)
        else:
            coords["lag"] = np.array([False, True])
            magnitudes = magnitudes.reshape(2, magnitudes.shape[0] // 2)
            if "node" not in coords and magnitudes.shape[0] > 0:
                coords["node"] = np.arange(magnitudes.shape[0])
            data_vars["magnitudes"] = (["lag", "node"], magnitudes)

    if time_series is not None:
        coords["time"] = np.arange(time_series.shape[0])
        if "node" not in coords and time_series.shape[1] > 0:
            coords["node"] = np.arange(time_series.shape[1])
        if "dim" not in coords and time_series.shape[2] > 0:
            coords["dim"] = np.arange(time_series.shape[2])
        data_vars["time_series"] = (
            ["time", "node", "dim"],
            time_series.data if hasattr(time_series, "data") else time_series,
        )

    if root_nodes is not None:
        if "node" not in coords and root_nodes.shape[0] > 0:
            coords["node"] = np.arange(root_nodes.shape[0])
        data_vars["root_nodes"] = (["node"], root_nodes)

    if time_lag == 0:
        data_vars["time_lag"] = ([], time_lag)

    ds = xr.Dataset(data_vars, coords=coords)
    return ds


def create_dynsys_dataset(
    *,
    adjacency_matrix: np.ndarray = None,
    time_series: np.ndarray = None,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Create an xarray Dataset containing all the components of the simulation.

    Parameters
    ----------
    adjacency_matrix : np.ndarray, optional
        The adjacency matrix representing the directed graph structure, shape (num_nodes, num_nodes).
    time_series : np.ndarray, optional
        The time series data for all nodes, shape (num_timesteps, num_nodes, node_dim).
    Returns
    -------
    xr.Dataset
        A dataset containing all components with appropriate coordinates.
    """
    if verbose:
        print(
            "Saving adjacency matrix of shape:",
            adjacency_matrix.shape if adjacency_matrix is not None else None,
        )
        print(
            "Saving time series of shape:",
            time_series.shape if time_series is not None else None,
        )

    # Create coordinates
    coords = {}
    data_vars = {}

    if adjacency_matrix is not None:
        coords["dim_in"] = np.arange(adjacency_matrix.shape[0])
        coords["dim_out"] = np.arange(adjacency_matrix.shape[1])
        data_vars["adjacency_matrix"] = (["dim_in", "dim_out"], adjacency_matrix)

    if time_series is not None:
        coords["time"] = np.arange(time_series.shape[0])
        if "system" not in coords and time_series.shape[1] > 0:
            coords["system"] = np.arange(time_series.shape[1])
        if "dim" not in coords and time_series.shape[2] > 0:
            coords["dim"] = np.arange(time_series.shape[2])
        data_vars["time_series"] = (
            ["time", "system", "dim"],
            time_series.data if hasattr(time_series, "data") else time_series,
        )

    ds = xr.Dataset(data_vars, coords=coords)
    return ds


def save_xr_dataset(dataset: xr.Dataset, path: str) -> None:
    """
    Save an xarray dataset to a netCDF or zarr file.

    This function saves the provided xarray Dataset to disk in either netCDF (.nc) or
    zarr (.zarr) format based on the file extension. It automatically creates any
    necessary parent directories if they don't exist.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to save. Should be a valid xarray Dataset containing simulation data
        such as adjacency matrices, time series, or other model components.
    path : str
        The path to save the dataset to. Must end with either '.nc' for netCDF format
        or '.zarr' for zarr format.

    Raises
    ------
    ValueError
        If the file extension is not supported (.nc or .zarr).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".nc":
        dataset.to_netcdf(path)
    elif path.suffix == ".zarr":
        dataset.to_zarr(path)
    else:
        raise ValueError(f"Unsupported file extension: {path}. Options are .nc, .zarr")


def load_xr_dataset(path: str) -> xr.Dataset:
    """
    Load an xarray dataset from a netCDF (.nc) or zarr (.zarr) file.

    Parameters
    ----------
    path : str
        The path to load the dataset from. Must end with either '.nc' for netCDF format
        or '.zarr' for zarr format.

    Returns
    -------
    xarray.Dataset
        The loaded dataset.
    """
    path = Path(path)
    if path.suffix == ".nc":
        return xr.open_dataset(path)
    elif path.suffix == ".zarr":
        return xr.open_zarr(path)
    else:
        raise ValueError(f"Unsupported file extension: {path}. Options are .nc, .zarr")
