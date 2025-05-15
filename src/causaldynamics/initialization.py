import torch

from causaldynamics.scm import get_root_nodes_mask
from causaldynamics.systems import drive_sin, solve_random_systems, solve_system
from causaldynamics.utils import allocate_elements_based_on_ratios


def initialize_system_and_driver(
    num_timesteps,
    num_nodes,
    init_ratios,
    system_name,
    node_dim=3,
    time_lag=0,
    device=None,
    make_trajectory_kwargs={},
):
    """
    Initialize a system with a combination of dynamical systems and driving functions.

    This function creates initial values for a causal dynamical system by combining
    outputs from dynamical systems (specified by system_name) with sinusoidal and
    linear driving functions. The nodes are randomly permuted to ensure diversity.

    Parameters
    ----------
    num_timesteps : int
        Number of time steps to generate.
    num_nodes : int
        Total number of nodes in the graph.
    init_ratios : list or tuple
        Ratios for allocating nodes to different initialization types:
        [system_ratio, periodic_ratio].
    system_name : str
        Name of the dynamical system to use. If "random", uses random systems.
    node_dim : int, optional
        Dimension of each node, default is 3.
    time_lag : int, optional
        Number of time lag steps to include, default is 0 meaning no time lag.
    device : torch.device, optional
        Device to place the generated tensors on. If None, uses the default device.
    make_trajectory_kwargs : dict, optional
        Additional keyword arguments to pass to the system solver functions.

    Returns
    -------
    torch.Tensor
        Initialized tensor of shape (num_timesteps, num_nodes, node_dim) for time_lag=0,
        or (num_timesteps, 2*num_nodes, node_dim) when time_lag > 0.

    Notes
    -----
    When time_lag > 0, the function generates additional time steps and then
    concatenates the current time steps with future time steps to create a tensor
    that includes both current and lagged values.
    """
    with torch.no_grad():
        T = num_timesteps
        if time_lag and time_lag > 0:  # Calculate time_lag steps more
            T = T + time_lag
        N = num_nodes
        D = node_dim

        n_sys, n_sin = allocate_elements_based_on_ratios(N, init_ratios)

        if system_name == "random":
            d_sys = solve_random_systems(
                T, n_sys, make_trajectory_kwargs=make_trajectory_kwargs
            )
        else:
            d_sys = solve_system(
                T, n_sys, system_name, make_trajectory_kwargs=make_trajectory_kwargs
            )

        d_sin = drive_sin(T, n_sin, D, device=device)

        init = torch.cat((d_sys, d_sin), dim=1)
        idx = torch.randperm(init.shape[1], device=device)
        init = init[:, idx].contiguous()

        if time_lag and time_lag > 0:
            init_now = init[:num_timesteps, :, :]
            init_future = init[time_lag:T, :, :]
            init = torch.cat((init_now, init_future), dim=1)

        return init


def initialize_x(
    init: torch.Tensor,
    A: torch.Tensor,
    standardize: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Initialize a tensor for propagation by setting values for root nodes from initial values.
    Non-root nodes are initialized to zero.

    Parameters
    ----------
    init : torch.Tensor
        Initial values tensor of shape (num_timesteps, num_nodes, node_dim).
    A : torch.Tensor
        Adjacency matrix of shape (num_nodes, num_nodes) or (2*num_nodes, num_nodes)
        in case of time lag edges. A[i, j] = 1 indicates an edge from node i to node j.
    standardize : bool, optional
        Whether to internally standardize the output tensor after initialization.
        Default is False.
    device : torch.device or None, optional
        Device to place the output tensor on. Default is None.

    Returns
    -------
    torch.Tensor
        Initialized tensor of same shape as init, with values set for root nodes
        and zero for non-root nodes.

    Notes
    -----
    Root nodes are nodes with no incoming edges. For time lag edges, root nodes
    are determined separately for the current time step and past time steps.
    If standardization is enabled, the function adds Gaussian noise to the tensor
    and normalizes it to have zero mean and unit variance.
    """
    x = torch.zeros_like(init, device=device)

    # Case: No time lag edges
    if A.shape[0] == A.shape[1]:
        root_nodes = get_root_nodes_mask(A)
        x[:, root_nodes, :] = init[:, root_nodes, :]
    # Case: Time lag edges
    else:
        A_past = A[A.shape[0] // 2 :]
        A_now = A[: A.shape[0] // 2]

        root_nodes_now = get_root_nodes_mask(A_now)
        root_nodes_past = get_root_nodes_mask(A_past)
        root_nodes = torch.cat((root_nodes_now, root_nodes_past), dim=0)
        x[:, root_nodes, :] = init[:, root_nodes, :]

    if standardize:
        # Standardize over the time dimension
        x_mean = x.mean(dim=0)
        x_var = x.var(dim=0)
        x = (x - x_mean) / torch.sqrt(x_var)
        
    return x


def initialize_biases(
    num_nodes: int, node_dim: int, device: torch.device = None
) -> torch.Tensor:
    """
    Initialize biases for neural network nodes using standard normal distribution.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the network
    node_dim : int
        Number of features per node
    device : torch.device or None, optional
        Device to create the tensor on, default is None

    Returns
    -------
    torch.Tensor
        Tensor of shape (num_nodes, node_dim) containing the initialized biases
        sampled from a standard normal distribution N(0,1)
    """
    return torch.randn(num_nodes, node_dim, device=device)


def initialize_weights(
    num_nodes: int, node_dim: int, p_zero: float = 0.0, device: torch.device = None
) -> torch.Tensor:
    """
    Initialize weights for neural network nodes using standard normal distribution.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the network
    node_dim : int
        Number of dimensions per node (input and output dimensions)
    p_zero : float, optional
        Probability of setting a weight to zero for sparsity, default is 0.0
    device : torch.device or None, optional
        Device to create the tensor on, default is None

    Returns
    -------
    torch.Tensor
        Tensor of shape (num_nodes, node_dim, node_dim) containing
        the initialized weights with N(0,1) distribution
    """
    with torch.no_grad():
        weights = torch.empty(num_nodes, node_dim, node_dim, device=device)
        weights.normal_(0, 1)

        if p_zero > 0:
            mask = torch.rand_like(weights) > p_zero
            weights = weights * mask

        return weights
