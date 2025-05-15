import inspect
import random
from typing import Union

import dysts.flows as flows
import numpy as np
import torch
from joblib import Parallel, delayed

_DYSTS_3D_CHAOTIC_SYSTEMS = [
    "Lorenz",
    "LorenzBounded",
    "Lorenz84",
    "Rossler",
    "Thomas",
    "ThomasLabyrinth",
    "GlycolyticOscillation",
    "GuckenheimerHolmes",
    "Halvorsen",
    "Chua",
    "MultiChua",
    "Duffing",
    "DoubleGyre",
    "BlinkingRotlet",
    "LidDrivenCavityFlow",
    "BlinkingVortex",
    "InteriorSquirmer",
    "OscillatingFlow",
    "BickleyJet",
    "ArnoldBeltramiChildress",
    "JerkCircuit",
    "ForcedBrusselator",
    "WindmiReduced",
    "MooreSpiegel",
    "CoevolvingPredatorPrey",
    "KawczynskiStrizhak",
    "BelousovZhabotinsky",
    "IsothermalChemical",
    "VallisElNino",
    "RabinovichFabrikant",
    "NoseHoover",
    "Dadras",
    "RikitakeDynamo",
    "PehlivanWei",
    "SprottTorus",
    "SprottJerk",
    "SprottA",
    "SprottB",
    "SprottC",
    "SprottD",
    "SprottE",
    "SprottF",
    "SprottG",
    "SprottH",
    "SprottI",
    "SprottJ",
    "SprottK",
    "SprottL",
    "SprottM",
    "SprottN",
    "SprottO",
    "SprottP",
    "SprottQ",
    "SprottR",
    "SprottS",
    "SprottMore",
    "Arneodo",
    "Coullet",
    "Rucklidge",
    "Sakarya",
    "LiuChen",
    "RayleighBenard",
    "Finance",
    "Bouali2",
    "Bouali",
    "LuChenCheng",
    "LuChen",
    "QiChen",
    "ZhouChen",
    "BurkeShaw",
    "Chen",
    "ChenLee",
    "WangSun",
    "YuWang",
    "YuWang2",
    "SanUmSrisuchinwong",
    "DequanLi",
    "PanXuZhou",
    "Tsucs2",
    "NewtonLiepnik",
    "SaltonSea",
    "ExcitableCell",
    "CaTwoPlus",
    "FluidTrampoline",
    "Aizawa",
    "AnishchenkoAstakhov",
    "ShimizuMorioka",
    "GenesioTesi",
    "AtmosphericRegime",
    "Hadley",
    "ForcedVanDerPol",
    "ForcedFitzHughNagumo",
    "HindmarshRose",
    "Colpitts",
    "Laser",
    "Blasius",
    "TurchinHanski",
    "StickSlipOscillator",
    "HastingsPowell",
    "CellularNeuralNetwork",
    "BeerRNN",
    "Torus",
    "CaTwoPlusQuasiperiodic",
    "ItikBanksTumor",
]


def get_all_chaotic_system_names():
    """
    Get all chaotic system names available in the dysts.flows module.

    This function retrieves all class names from the dysts.flows module that represent
    dynamical systems, excluding base classes and special Python classes.

    Returns
    -------
    list
        A list of strings containing the names of all available chaotic systems
        in the dysts.flows module.
    """
    # Get all classes from the flows module, excluding base classes
    return [
        name
        for name, obj in vars(flows).items()
        if isinstance(obj, type)
        and not name.startswith("__")
        and name not in ["DynSys", "DynSysDelay"]
    ]


def get_all_chaotic_system_names_and_dimensions():
    """
    Get all chaotic system names and their corresponding dimensionality.

    This function retrieves all available chaotic systems from the dysts.flows module
    and determines the dimensionality of each system by solving it for a minimal
    number of timesteps and examining the shape of the resulting trajectory.

    Returns
    -------
    dict
        A dictionary mapping system names (str) to their dimensionality (int).
        Each key is a chaotic system name and each value is the number of
        dimensions in the system's state space.

    Notes
    -----
    The function solves each system for 2 timesteps with 1 node to determine
    the dimensionality, which is extracted from the shape of the resulting tensor.
    """
    dims = {}
    system_names = get_all_chaotic_system_names()
    for name in system_names:
        result = solve_system(2, 1, name)
        dimensionality = result.shape[2]
        dims[name] = dimensionality

    return dims


def solve_single_system(model, num_timesteps, make_trajectory_kwargs):
    """
    Solve a single dynamical system with randomized initial conditions.

    This function creates a copy of the provided model to avoid race conditions
    when running in parallel, randomizes the initial conditions, and generates
    a trajectory for the specified number of timesteps.

    Parameters
    ----------
    model : dysts.flows.DynSys
        The dynamical system model to solve
    num_timesteps : int
        Number of timesteps to simulate
    make_trajectory_kwargs : dict
        Keyword arguments to pass to the make_trajectory method of the system

    Returns
    -------
    numpy.ndarray
        Solution trajectory of the dynamical system
    """
    # Create a copy of the model to avoid race conditions
    model_copy = model.__class__()
    model_copy.ic = model_copy.ic * np.random.random()
    sol = model_copy.make_trajectory(num_timesteps, **make_trajectory_kwargs)

    return sol


def solve_system(
    num_timesteps: int,
    num_systems: int,
    system_name: str,
    make_trajectory_kwargs: dict = {},
) -> torch.Tensor:
    """
    Solve a dynamical system for multiple nodes in parallel.

    This function creates multiple instances of the specified dynamical system,
    each with randomized initial conditions, and generates trajectories for all nodes
    in parallel using joblib.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps to simulate
    num_systems : int
        Number of systems (separate, uncoupled instances) to simulate
    system_name : str
        Name of the dynamical system class from dysts.flows to use
    make_trajectory_kwargs : dict
        Keyword arguments to pass to the make_trajectory method of the system

    Returns
    -------
    torch.Tensor
        Tensor of shape (num_timesteps, num_systems, num_dimensions) containing
        the solution trajectories for all nodes
    """
    system = getattr(flows, system_name)
    model = system()

    # Use joblib to parallelize the computation
    data = Parallel(n_jobs=-1)(
        delayed(solve_single_system)(model, num_timesteps, make_trajectory_kwargs)
        for _ in range(num_systems)
    )

    try:
        data = torch.from_numpy(np.array(data)).float()

        # (num_timesteps, num_systems, num_dimensions) -> (num_systems, num_timesteps, num_dimensions)
        data = data.permute(1, 0, 2)

        return data
    except Exception as e:
        print(f"Error solving system {system_name}: {e}. Skipping system.")
        return None


def count_distribution_of_system_by_dimensions():
    """
    Count the distribution of chaotic systems by their dimensions.

    This function retrieves all chaotic system names and their dimensions,
    then counts how many systems exist for each dimension.

    Returns
    -------
    dict
        A dictionary where keys are dimensions (int) and values are the count
        of systems (int) with that dimension, sorted by dimension.
    """
    # Count the number of systems per dimension
    dims = get_all_chaotic_system_names_and_dimensions()
    dimension_counts = {}
    for system, dim in dims.items():
        if dim in dimension_counts:
            dimension_counts[dim] += 1
        else:
            dimension_counts[dim] = 1

    # Sort by dimension
    sorted_counts = dict(sorted(dimension_counts.items()))
    return sorted_counts


def get_3d_chaotic_system_names():
    """
    Get the names of all 3D chaotic systems available in the dysts.flows module.

    Returns
    -------
    list
        A list of strings containing the names of all available 3D chaotic systems
        in the dysts.flows module.
    """
    return _DYSTS_3D_CHAOTIC_SYSTEMS


def solve_random_systems(
    num_timesteps: int,
    num_systems: int,
    make_trajectory_kwargs: dict = {},
    max_retry: int = 10,
) -> torch.Tensor:
    """
    Solve multiple random chaotic systems for the given number of timesteps and nodes.

    This function randomly selects 3D chaotic systems from the available systems in dysts.flows,
    solves them for the specified number of timesteps, and returns the combined trajectories.
    If a system fails to integrate (particularly SprottJerk), it attempts to use an alternative system.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps to simulate for each system
    num_systems : int
        Number of systems to simulate
    make_trajectory_kwargs : dict
        kwargs passed on the dysys.make_trajectory. Default is an empty dict.
    max_retry : int
        Maximum number of retries for a system to integrate before raising an exception

    Returns
    -------
    torch.Tensor
        Tensor of shape (num_timesteps, num_systems, 3) containing
        the solution trajectories for all nodes
    """
    systems_dim3 = get_3d_chaotic_system_names()
    # Randomly select N different attractors from systems_dim3
    selected_systems = random.sample(systems_dim3, min(num_systems, len(systems_dim3)))

    # Create model instances for each selected system
    models = [getattr(flows, system)() for system in selected_systems]

    # Process each model using solve_single_system
    data = []
    for i in range(num_systems):
        sol = None
        counter = 0
        while sol is None:
            system = random.choice(
                [
                    s
                    for s in systems_dim3
                    if s != selected_systems[i % len(selected_systems)]
                ]
            )
            alternative_model = getattr(flows, system)()
            sol = solve_single_system(
                alternative_model,
                num_timesteps,
                make_trajectory_kwargs=make_trajectory_kwargs,
            )
            counter += 1
            if counter > max_retry:
                raise Exception(
                    f"Failed to integrate system {selected_systems[i % len(selected_systems)]} after 10 attempts."
                )

        data.append(sol)
    data = torch.from_numpy(np.array(data)).float()
    data = data.permute(1, 0, 2)

    return data

def get_adjacency_matrix_from_jac(dyn_sys: Union[str, flows.DynSys]):
    """
    Extract the adjacency matrix from the Jacobian of a dynamical system.

    The adjacency matrix represents the structure of interactions between variables
    in the dynamical system. A value of 1 at position (i,j) indicates that variable j
    affects the evolution of variable i.

    Parameters
    ----------
    dyn_sys : str or dysts.flows.DynSys
        The dynamical system class name (as string) or instance with a _jac method.
        If _jac is None or not present, the function returns None.

    Returns
    -------
    numpy.ndarray or None
        Binary adjacency matrix where 1 indicates a non-zero entry in the Jacobian.
        Returns None if the dynamical system doesn't have a _jac method.

    Notes
    -----
    The function evaluates the Jacobian at a default point (1,1,1,t=0) with all
    system parameters set to 1. This is sufficient to determine the structural
    connectivity of the system.
    """
    if isinstance(dyn_sys, str):
        dyn_sys = getattr(flows, dyn_sys)

    # Check if the dynamical system has a Jacobian method
    if not hasattr(dyn_sys, "_jac") or dyn_sys._jac is None:
        return None

    # Check if the system is 3D by examining the signature of _jac
    sig = inspect.signature(dyn_sys._jac)
    params = list(sig.parameters.keys())

    # First 4 parameters are x, y, z, t with values 1, 1, 1, 0
    # All remaining parameters are set to 1
    num_params = len(sig.parameters)
    args = [1, 1, 1, 0] + [1] * (num_params - 4)
    jac = dyn_sys._jac(*args)

    # Convert to numpy array
    jac_array = np.array(jac)

    # Create binary adjacency matrix (1 where Jacobian is non-zero)
    adj_mat = np.zeros_like(jac_array, dtype=int)
    adj_mat[jac_array != 0] = 1

    return adj_mat


def drive_sin(num_timesteps, num_nodes, node_dim, max_num_periods=10, device=None):
    """
    Generate time series data with sinusoidal dynamics.

    This function creates synthetic time series data where each node follows sinusoidal
    patterns with randomly generated parameters. Each node can have multiple dimensions,
    and each dimension will have its own unique sinusoidal pattern with different
    amplitude, phase shift, and frequency.

    The sinusoidal pattern follows the formula: amplitude * sin(t + phase_shift),
    where t varies from 0 to a randomly chosen maximum time value for each node and dimension.

    Parameters
    ----------
    num_timesteps : int
        Number of time steps to generate.
    num_nodes : int
        Number of nodes in the system.
    node_dim : int
        Dimension of each node. Each dimension will have its own sinusoidal pattern.
    max_num_periods : int, optional
        Maximum number of periods to include in the time series, default is 10.
        This controls the maximum frequency of the sinusoidal patterns.
    device : torch.device, optional
        Device to place the generated tensors on. If None, uses the default device.

    Returns
    -------
    torch.Tensor
        Generated time series data with shape (time, num_nodes, node_dim).
        Each node-dimension combination follows a unique sinusoidal pattern.

    Notes
    -----
    - Amplitudes are randomly sampled from a uniform distribution in [-1, 1]
    - Phase shifts are randomly sampled from a uniform distribution in [0, 2π]
    - The maximum time value for each node-dimension is randomly sampled, resulting
      in different frequencies for different node-dimensions
    """
    with torch.no_grad():
        amplitude = (
            2 * torch.rand((num_nodes, node_dim), device=device) - 1
        )  # random uniform in [-1, 1]
        phase_shift = (
            2 * np.pi * torch.rand((num_nodes, node_dim), device=device)
        )  # random uniform in [0, 2π]
        data = torch.zeros((num_timesteps, num_nodes, node_dim), device=device)

        if num_nodes > 0:
            max_time = (
                max_num_periods * 2 * torch.rand((num_nodes, node_dim), device=device)
            )  # random uniform in [0, 10*2π]
            time_points = torch.from_numpy(
                np.linspace(0, max_time.numpy(), num_timesteps)
            )
            for i, t in enumerate(time_points):
                data[i, :, :] = amplitude * torch.sin(t + phase_shift)

        return data
