import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from jsonargparse import CLI

from causaldynamics.data_io import create_output_dataset, save_xr_dataset
from causaldynamics.initialization import (
    initialize_biases,
    initialize_system_and_driver,
    initialize_weights,
    initialize_x,
)
from causaldynamics.mlp import calculate_magnitudes, propagate_mlp
from causaldynamics.plot import animate_3d_trajectories, plot_scm, plot_trajectories, plot_3d_trajectories
from causaldynamics.scm import (
    calculate_all_adjacency_matrices,
    create_scm_graph,
    create_time_lag_adj_mat,
    flatten_time_lag_adj_mat,
    get_root_nodes_mask,
    sample_scale_free_DAG,
)
from causaldynamics.utils import get_timestamp, set_rng_seed

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def setup_environment(
    seed: int, out_dir: str, add_timestamp: bool, add_outdir_note: str = None
) -> Path:
    """
    Set up the environment including RNG seed and output directory.

    Parameters
    ----------
    seed : int
        Random number generator seed for reproducibility
    out_dir : str
        Base output directory path
    add_timestamp : bool
        If True, append a timestamp to the output directory path
    add_outdir_note : str, optional
        If provided, append this string as a subdirectory to the output path

    Returns
    -------
    Path
        The complete output directory path that was created

    Notes
    -----
    This function:
    - Sets matplotlib animation embed limit to 50MB
    - Seeds the random number generators
    - Creates the output directory with optional timestamp and note
    """
    mpl.rcParams["animation.embed_limit"] = 50 * 1024**2  # 50MB
    set_rng_seed(seed)

    if add_timestamp:
        out_dir = Path(out_dir) / get_timestamp()
    if add_outdir_note:
        out_dir = out_dir / add_outdir_note
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {out_dir}")
    return Path(out_dir)

def create_scm(
    num_nodes: int,
    node_dim: int,
    confounders: bool,
    adjacency_matrix: np.ndarray = None,
    graph: str = "scale-free",
    time_lag: int = None,
    time_lag_edge_probability: float = 0.1,
    max_tries: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a Structural Causal Model (SCM) with specified parameters.

    This function generates a directed acyclic graph (DAG) representing a causal system,
    along with associated weights, biases, and other parameters. It ensures that the
    generated graph has at least one root node (node with no parents).

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the causal graph.
    node_dim : int
        Dimensionality of each node's state vector.
    confounders : bool
        Whether to allow confounders (common causes) in the graph structure.
    adjacency_matrix : numpy.ndarray, optional
        Pre-defined adjacency matrix to use. If None, a new one will be generated.
    graph : str, default='scale-free'
        Type of graph to generate. Options:
        - 'scale-free': Generate a scale-free DAG
        - 'all_uniform': Sample uniformly from all possible DAGs
    time_lag : int, optional
        Number of time steps for lagged effects. If provided and > 0,
        time-lagged edges will be added to the graph.
    time_lag_edge_probability : float, default=0.1
        Probability of adding time-lagged edges when time_lag > 0.
    max_tries : int, default=100
        Maximum number of attempts to generate a graph with at least one root node.

    Returns
    -------
    tuple
        A tuple containing:
        - A : numpy.ndarray
            Adjacency matrix representing the graph structure
            with shape (num_nodes, num_nodes) if time_lag is None or 0,
            or (2*num_nodes, num_nodes) if time_lag>0.
        - W : numpy.ndarray
            Weight matrices for each node with shape (num_nodes, node_dim, node_dim).
        - b : numpy.ndarray
            Bias vectors for each node with shape (num_nodes, node_dim).
        - root_nodes : numpy.ndarray
            Boolean mask indicating which nodes are root nodes with shape (num_nodes,).
        - magnitudes : numpy.ndarray
            Magnitudes of the weights for each node with shape (num_nodes,).

    Raises
    ------
    ValueError
        If no adjacency matrix with root nodes is found after max_tries attempts.

    Notes
    -----
    Root nodes are essential as they serve as exogenous variables in the causal system.
    The function will retry generating graphs until at least one root node is found
    or the maximum number of tries is reached.
    """
    logger.info(
        f"Creating SCM with {num_nodes} nodes and {node_dim} dimensions each..."
    )
    root_nodes_exist = False
    num_tries = 0
    while not root_nodes_exist and num_tries < max_tries:
        if adjacency_matrix is None:
            if graph == "scale-free":
                A = sample_scale_free_DAG(num_nodes, confounders=confounders)
            elif graph == "all_uniform":
                A_all = calculate_all_adjacency_matrices(
                    num_nodes=num_nodes, confounders=confounders
                )
                A = A_all[np.random.choice(len(A_all))]
        else:
            A = adjacency_matrix

        if time_lag and time_lag > 0 and time_lag_edge_probability > 0.0:
            A_lag = create_time_lag_adj_mat(A, time_lag_edge_probability)
            A_flat = flatten_time_lag_adj_mat(A_lag)
            W = initialize_weights(A_flat.shape[0], node_dim)
            b = initialize_biases(A_flat.shape[0], node_dim)
            root_nodes = get_root_nodes_mask(A_flat[: A_flat.shape[0] // 2])
            if sum(root_nodes) > 0:
                root_nodes_exist = True
                magnitudes = calculate_magnitudes(W)
                logger.debug(f"Found {sum(root_nodes)} root nodes")
                return A_lag, W, b, root_nodes, magnitudes
            else:
                num_tries += 1
        else:
            W = initialize_weights(num_nodes, node_dim)
            b = initialize_biases(num_nodes, node_dim)
            root_nodes = get_root_nodes_mask(A)
            if sum(root_nodes) > 0:
                root_nodes_exist = True
                magnitudes = calculate_magnitudes(W)
                logger.debug(f"Found {sum(root_nodes)} root nodes")
                return A, W, b, root_nodes, magnitudes
            else:
                num_tries += 1
    if not root_nodes_exist:
        raise ValueError(
            "No adjacency matrix with root nodes found in {max_tries} tries."
        )


def simulate_system(
    A,
    W,
    b,
    num_timesteps: int = None,
    num_nodes: int = None,
    init_ratios: list = [1.0, 0.0],
    system_name: str = None,
    make_trajectory_kwargs: dict = {},
    init: torch.Tensor = None,
    time_lag: int = None,
    standardize: bool = False,
) -> xr.DataArray:
    """
    Run the system simulation.

    This function simulates a dynamical system using either random systems or a specified
    system type.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix representing the directed graph structure.
    W : numpy.ndarray or torch.Tensor
        Weight matrices for each node in the MLP.
    b : numpy.ndarray or torch.Tensor
        Bias vectors for each node in the MLP.
    num_timesteps : int
        Number of timesteps to simulate.
    num_nodes : int
        Number of nodes in the system.
    init_ratios: list
        List of root node driver system ratios for chaotic, sin, and linear driver systems.
        [2, 1] means that twice as many chaotic driver as sin drivers are initialized randomly.
        Not used if init is provided!
    system_name : str, optional
        Name of the dynamical system to simulate. Use "random" for random systems.
        If init is not None, this parameter is ignored.
    init : torch.Tensor, optional
        Initial values for the system. If None, random initial values are used
        based on the system_name.
    time_lag : int, optional
        Time lag for the system. If None, no time lag is used.
    standardize : bool, optional
        Whether to standardize the trajectories creating an iSCM.

    Returns
    -------
    xarray.DataArray
        A data array containing the simulated trajectories with dimensions ["time", "node", "dim"].

    """
    logger.info(f"Simulating {system_name} system for {num_timesteps} timesteps...")

    if A.ndim == 3:
        A = flatten_time_lag_adj_mat(A)

    if init is None:
        init = initialize_system_and_driver(
            num_timesteps=num_timesteps,
            num_nodes=num_nodes,
            init_ratios=init_ratios,
            system_name=system_name,
            time_lag=time_lag,
            make_trajectory_kwargs=make_trajectory_kwargs,
        )

        init = initialize_x(init, A, standardize=standardize)

    x = propagate_mlp(A=A, W=W, b=b, init=init, standardize=standardize)
    da = xr.DataArray(x, dims=["time", "node", "dim"])

    # if time lag, only return the first node values. The others are just shifted
    if A.shape[0] != A.shape[1]:
        da = da.isel(node=slice(0, A.shape[1]))
    return da


def create_plots(
    da,
    A,
    root_nodes: torch.Tensor = None,
    show_plot: bool = False,
    save_plot: bool = False,
    return_html_anim: bool = False,
    create_animation: bool = False,
    out_dir: Path = None,
    seed: int = None,
) -> None:
    """
    Generate and save all visualizations for the causal dynamical system.

    This function creates and optionally saves various visualizations of the
    causal dynamical system, including the structural causal model graph,
    trajectory plots, and 3D animations.

    Parameters
    ----------
    da : xarray.DataArray
        The time series data with dimensions ['time', 'node', 'dim'].
    A : numpy.ndarray
        The adjacency matrix representing the causal structure.
    root_nodes : torch.Tensor, optional
        Boolean tensor indicating which nodes are root nodes (have no parents).
        If None, will be automatically determined from the adjacency matrix.
    show_plot : bool, optional
        Whether to display the plots interactively. Default is False.
    save_plot : bool, optional
        Whether to save the plots to disk. Default is False.
    return_html_anim : bool, optional
        Whether to return HTML animation for Jupyter notebooks. Default is False.
    create_animation : bool, optional
        Whether to create 3D animations of the trajectories. Default is False.
    out_dir : Path, optional
        Directory path where visualizations will be saved. Default is current directory.
    seed : int, optional
        Random seed used for the simulation, used in filenames for saved plots.

    Returns
    -------
    None
        The function saves visualizations to disk and/or displays them based on the provided options.

    Notes
    -----
    This function creates three types of visualizations:
    1. A structural causal model (SCM) graph showing causal relationships
    2. Time series plots of node trajectories
    3. Optional 3D animations of the system dynamics

    If both show_plot and save_plot are False, the visualizations will still be generated
    but not displayed or saved.
    """
    if out_dir is None:
        out_dir = Path(".")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Generating visualizations")
    # Plot SCM graph
    G = create_scm_graph(A)
    if root_nodes is None:
        root_nodes = get_root_nodes_mask(A)
    plot_scm(G=G, root_nodes=root_nodes)
    if save_plot:
        plot_path = (
            out_dir / f"scm_graph_seed{seed}.png"
            if seed
            else out_dir / f"scm_graph.png"
        )
        plt.savefig(plot_path)
        logger.debug(f"Saved SCM graph to {plot_path}")
    if show_plot:
        plt.show()
    plt.close()

    # Plot trajectories
    plot_trajectories(da, root_nodes=root_nodes, sharey=False)
    if save_plot:
        plot_path = (
            out_dir / f"trajectories_seed{seed}.png"
            if seed
            else out_dir / "trajectories.png"
        )
        plt.savefig(plot_path)
        logger.debug(f"Saved trajectories plot to {plot_path}")
    if show_plot:
        plt.show()
    plt.close()

    # Plot 3D trajectories
    plot_3d_trajectories(da, root_nodes=root_nodes)
    if save_plot:
        plot_path = (
            out_dir / f"3D_trajectories_seed{seed}.png"
            if seed
            else out_dir / "3D_trajectories.png"
        )
        plt.savefig(plot_path)
        logger.debug(f"Saved 3D trajectories plot to {plot_path}")
    if show_plot:
        plt.show()
    plt.close()

    # Animate trajectories
    if save_plot:
        save_path = (
            out_dir / f"animation_seed{seed}.mp4"
            if seed
            else out_dir / f"animation.mp4"
        )
    if create_animation:
        logger.info("Creating 3D animation of trajectories...")
        animate_3d_trajectories(
            da,
            frame_skip=5,
            rotation_speed=0.2,
            rotate=True,
            show_history=True,
            plot_type="subplots",
            root_nodes=root_nodes,
            save_path=save_path,
            return_html_anim=return_html_anim,
            show_plot=show_plot,
        )
        if save_path:
            logger.debug(f"Saved animation to {save_path}")

    plt.close()


def create(
    seed: int = None,
    num_nodes: int = 5,
    num_timesteps: int = 300,
    node_dim: int = 3,
    scm_confounders: bool = False,
    standardize: bool = False,
    graph: str = "scale-free",
    system_name: str = "Lorenz",
    make_trajectory_kwargs: dict = {},
    init_ratios: list = [1.0, 0.0],
    init: torch.Tensor = None,
    time_lag: int = None,
    time_lag_edge_probability: float = 0.1,
    out_dir_base: str = "output",
    add_outdir_timestamp: bool = True,
    add_outdir_note: str = None,
    out_dir_data: str = "data",
    out_dir_plots: str = "plots",
    plot: bool = True,
    show_plot: bool = False,
    save_plot: bool = True,
    create_animation: bool = False,
    return_html_anim: bool = False,
    save_data: bool = True,
) -> xr.DataArray:
    """
    Create and simulate a Structural Causal Model (SCM) with dynamic systems.

    This function orchestrates the entire SCM simulation pipeline, including system creation,
    simulation, visualization, and data saving.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. Default is None.
    num_nodes : int, optional
        Number of nodes in the SCM graph. Default is 5.
    num_timesteps : int, optional
        Number of timesteps to simulate. Default is 300.
    node_dim : int, optional
        Dimension of each node. Default is 3.
    scm_confounders : bool, optional
        Whether to include confounders in the SCM. Default is False.
    standardize : bool, optional
        Whether to standardize the trajectories. Default is False.
    graph : str, optional
        Type of graph to generate ('scale-free' or 'all_uniform'). Default is 'scale-free'.
    system_name : str, optional
        Name of the dynamical system to use. Default is "Lorenz".
    make_trajectory_kwargs : dict, optional
        Additional arguments for trajectory generation. Default is {}.
    init_ratios : list, optional
        Ratios for different types of node initializations [system, periodic]. Default is [1., 0.].
    init : torch.Tensor, optional
        Initial values for the system. If None, generated based on init_ratios. Default is None.
    time_lag : int, optional
        Time lag for the system. If None, no time lag is used. Default is None.
    time_lag_edge_probability : float, optional
        Probability of creating time lag edges. Default is 0.1.
    out_dir_base : str, optional
        Base output directory. Default is "output".
    add_outdir_timestamp : bool, optional
        Whether to add a timestamp to the output directory. Default is True.
    add_outdir_note : str, optional
        Additional note to add to the output directory name. Default is None.
    out_dir_data : str, optional
        Subdirectory for saving data. Default is "data".
    out_dir_plots : str, optional
        Subdirectory for saving plots. Default is "plots".
    plot : bool, optional
        Whether to generate plots. Default is True.
    show_plot : bool, optional
        Whether to display plots. Default is False.
    save_plot : bool, optional
        Whether to save plots to disk. Default is True.
    create_animation : bool, optional
        Whether to create 3D animations. Make sure, ffmpeg is installed. Default is False.
    return_html_anim : bool, optional
        Whether to return HTML animations. Default is False.
    save_data : bool, optional
        Whether to save simulation data. Default is True.

    Returns
    -------
    xarray.DataArray
        The simulated time series data with dimensions ["time", "node", "dim"].

    Notes
    -----
    The function performs the following steps:
    1. Sets up the environment with the specified seed and output directories
    2. Creates an SCM with the specified parameters
    3. Simulates the system dynamics
    4. Generates visualizations if requested
    5. Saves the simulation data if requested
    """
    # Setup
    logger.info("Starting SCM simulation...")
    out_dir = setup_environment(
        seed, out_dir_base, add_outdir_timestamp, add_outdir_note
    )
    # Create and simulate system

    logger.info("Creating SCM system...")
    A, W, b, root_nodes, magnitudes = create_scm(
        num_nodes,
        node_dim,
        confounders=scm_confounders,
        graph=graph,
        time_lag=time_lag,
        time_lag_edge_probability=time_lag_edge_probability,
    )

    logger.info("Simulating system dynamics...")
    da = simulate_system(
        A,
        W,
        b,
        num_timesteps=num_timesteps,
        num_nodes=num_nodes,
        init_ratios=init_ratios,
        init=init,
        system_name=system_name,
        time_lag=time_lag,
        make_trajectory_kwargs=make_trajectory_kwargs,
        standardize=standardize,
    )

    # Generate plots only if plot is True
    if plot:
        logger.info("Generating plots and visualizations...")
        create_plots(
            da,
            A,
            root_nodes=root_nodes,
            seed=seed,
            out_dir=Path(out_dir) / out_dir_plots,
            show_plot=show_plot,
            save_plot=save_plot,
            return_html_anim=return_html_anim,
            create_animation=create_animation,
        )

    # Save data
    if save_data:
        logger.info("Saving simulation data...")
        dataset = create_output_dataset(
            adjacency_matrix=A,
            weights=W,
            biases=b,
            magnitudes=magnitudes,
            time_series=da,
            root_nodes=root_nodes,
            time_lag=time_lag,
            verbose=False,
        )
        data_path = (
            Path(out_dir)
            / out_dir_data
            / f"S{system_name}_N{num_nodes}_T{num_timesteps}_seed{seed}.nc"
        )
        save_xr_dataset(dataset, data_path)
        logger.info(f"Data saved to {data_path}")
    logger.info("Simulation completed successfully! :)")
    return da


if __name__ == "__main__":
    CLI(create)
