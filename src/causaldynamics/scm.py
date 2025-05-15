import networkx as nx
import numpy as np
import torch


class GrowingNetworkWithRedirection:
    """
    Generate a temporal growing network with redirection graph.
    Following Krapivsky et al. 2001.
    """

    def __init__(self, num_nodes: int, redirect_prob: float = None):
        """
        Initialize the GNR class.

        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        redirect_prob : float, optional
            Probability of adding a node. Default is None.
        """

        if redirect_prob is not None and (redirect_prob < 0.0 or redirect_prob > 1.0):
            raise ValueError(
                f"The probability 'redirect_prob' must be between 0. and 1, got {redirect_prob}!"
            )

        if num_nodes < 1:
            raise ValueError(
                f"The number of nodes 'num_nodes' must be greater than 0, got {num_nodes}!"
            )

        self.n = num_nodes
        self.r = redirect_prob
        self.G = None

        self.A_k = None  # Attachment kernel
        self.ancestors = None  # Keep track of ancestors

    def generate(self) -> torch.Tensor:
        """
        Generate a growing network with redirection.

        This method initializes the graph tensor `G`, the attachment kernel `A_k`
        and the ancestors tensor.
        It then adds the initial node to the graph and iteratively adds the remaining
        nodes based on the redirection probability `r`.

        Returns
        -------
        torch.Tensor
            A tensor representing the generated temporal graph with shape (n, n).
        """

        self.G = torch.zeros(self.n, self.n)
        self.A_k = torch.zeros(self.n)
        self.ancestors = torch.zeros(self.n, dtype=torch.int32)

        if self.n == 1:
            self.G = torch.tensor([[0]])
            self.A_k = torch.tensor([0])
            self.ancestors = torch.tensor([0])
            return self.G

        self._add_initial_nodes_to_graph()

        if self.r is not None and self.r > 0:
            # Draw n uniform random numbers that decide
            # whether to add a node or redirect it to an ancestor
            redirect = torch.rand(self.n - 1) < self.r
            for i, r in enumerate(redirect):
                self._add_node_to_graph(i + 1, redirect=r)
        else:
            # Add n-2 nodes
            for i in range(self.n - 1):
                self._add_node_to_graph(i + 1, redirect=False)

        return self.G

    def generate_with_confounders(self):
        """
        Generate a growing network with redirection and confounders.

        This method creates two separate growing networks using the standard generate
        method, then combines them to create a network with confounding effects.
        The second network is rotated to create different connection patterns,
        and the combined network is thresholded to ensure binary edges.

        The method ensures that the resulting graph maintains a directed acyclic
        structure by zeroing out the upper triangle and the diagonal of the adjacency matrix.

        Returns
        -------
        torch.Tensor
            A tensor representing the generated temporal graph with confounders,
            with shape (n, n).
        """
        G1 = self.generate()
        G2 = self.generate()

        G2 = torch.rot90(G2, k=1, dims=(0, 1))
        G = G1 + G2

        # replace all values greater than 1 with 1
        G = torch.where(G > 1, torch.ones_like(G), G)

        # Ensure the upper right triangle is zero
        mask = torch.triu(torch.ones_like(G), diagonal=0)
        G = G * (1 - mask)

        return G

    def _add_initial_nodes_to_graph(self) -> None:
        """
        Add the initial two nodes and one edge to the graph.

        This method initializes the first node in the graph tensor `G`, sets its
        attachment kernel values in `A_k`, and records it as its own ancestor.

        The initial node is added to the last two time steps of the graph tensor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.G[0, 0] = 0
        self.G[1, 0] = 1
        self.A_k[0] = 1
        self.A_k[1] = 0
        self.ancestors[0] = 0
        self.ancestors[1] = 0

    def _add_node_to_graph(self, i: int, *, redirect: bool = False) -> None:
        """
        Add a node to the graph.

        This method adds a new node to the graph tensor `G` at the specified index `i`.
        The node is either added as a new node or redirected to an ancestor based on the
        `redirect` flag. The method updates the attachment kernel values in `A_k`,
        and records the ancestor of the new node.

        Parameters
        ----------
        i : int
            The index of the node to be added.
        redirect : bool, optional
            If True, the node is redirected to an ancestor. Default is False.

        Returns
        -------
        None
        """
        A_k = self.A_k[:i]
        A_k_prob = A_k / A_k.sum()
        chosen_idx = torch.multinomial(A_k_prob, 1).item()

        # If redirect is True, the node is redirected to the ancestor node
        if redirect:
            chosen_idx = self.ancestors[chosen_idx]

        self.G[i, chosen_idx] = 1

        # Add node to chosen node
        self.A_k[chosen_idx] += 1
        self.ancestors[i] = chosen_idx

        # Add node from previous time step
        self.A_k[i] += 1


# Alias for GrowingNetworkWithRedirection
GNR = GrowingNetworkWithRedirection


def sample_scale_free_DAG(
    num_nodes: int, redirect_prob: float = 0.0, confounders=False
) -> torch.Tensor:
    """
    Sample a Krapivsky adjacency matrix.

    Parameters
    ----------
    num_nodes : int
        The number of nodes in the graph.
    redirect_prob : float, optional
        The probability of redirecting the new node to an ancestor. Default is 0.
    confounders : Whether to add scale-free confounders

    Returns
    -------
    torch.Tensor
        The adjacency matrix of shape (num_nodes, num_nodes)
    """
    if confounders:
        return GNR(num_nodes, redirect_prob).generate_with_confounders()
    else:
        return GNR(num_nodes, redirect_prob).generate()


def create_scm_graph(
    adjacency_matrix: torch.Tensor | np.ndarray,
    add_root_nodes_self_edges: bool = True,
) -> nx.DiGraph:
    """
    Create a NetworkX DiGraph from an adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : torch.Tensor or numpy.ndarray
        The adjacency matrix A of shape (n, n) or (2, n, n)
        where A[i,j] indicates an edge from i to j in the first case
        and A[0,i,j] and A[1,i,j] indicate a current and past edge
        from i to j in the second case.

    Returns
    -------
    nx.DiGraph
        A directed graph representing the SCM
    """
    A = adjacency_matrix

    n = A.shape[1]
    G = nx.MultiDiGraph()

    # Add nodes
    for i in range(n):
        G.add_node(i, label=f"Node {i}")

    # Add edges
    if A.ndim == 3:
        A_now = A[0]
        A_past = A[1]
        if add_root_nodes_self_edges:
            A_now = add_root_self_edges(A_now)
            A_past = add_root_self_edges(A_past)
        # Convert to numpy if it's a torch tensor
        if isinstance(A_now, torch.Tensor):
            A_now = A_now.detach().cpu().numpy()
        if isinstance(A_past, torch.Tensor):
            A_past = A_past.detach().cpu().numpy()

        edge_indices = np.argwhere(A_now)
        for idx in range(edge_indices.shape[0]):
            src, dst = edge_indices[idx]
            G.add_edge(int(src), int(dst))

        edge_indices = np.argwhere(A_past)
        for idx in range(edge_indices.shape[0]):
            src, dst = edge_indices[idx]
            G.add_edge(int(src), int(dst), label="lagged")
    else:
        if add_root_nodes_self_edges:
            A = add_root_self_edges(A)
        if isinstance(A, torch.Tensor):
            A = A.detach().cpu().numpy()
        edge_indices = np.argwhere(A)
        for idx in range(edge_indices.shape[0]):
            src, dst = edge_indices[idx]
            G.add_edge(int(src), int(dst))
    return G


def get_root_nodes_mask(A: torch.Tensor) -> torch.Tensor:
    """
    Identify root nodes in a directed graph represented by an adjacency matrix.

    Root nodes are defined as nodes that have no incoming edges.

    Parameters
    ----------
    A : torch.Tensor
        Adjacency matrix of shape (n, n) where A[i,j] indicates an edge from i to j

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (n,) where True values indicate root nodes
    """
    if A.ndim == 3:
        A_sum = A.sum(dim=0)
        return ~A_sum.T.bool().any(dim=1)
    else:
        return ~A.T.bool().any(dim=1)


def get_root_nodes(A: torch.Tensor) -> torch.Tensor:
    """
    Find root nodes in a graph (nodes with no incoming edges).

    Parameters
    ----------
    A : torch.Tensor
        The adjacency matrix of shape (n, n) where A[i,j] indicates an edge from i to j

    Returns
    -------
    torch.Tensor
        Indices of root nodes
    """
    return torch.where(get_root_nodes_mask(A))[0]


def get_nodes_by_in_degree(
    A: torch.Tensor,
    sort: bool = True,
    exclude_root_nodes: bool = True,
    return_num_in_edges: bool = False,
) -> torch.Tensor:
    """
    Get nodes ordered by their number of incoming edges.

    Parameters
    ----------
    A : torch.Tensor
        The adjacency matrix of shape (n, n) where A[i,j] indicates an edge from i to j
    sort : bool, optional
        If True, sort nodes by number of incoming edges in descending order. Default is True.
    exclude_root_nodes : bool, optional
        If True, exclude nodes with no incoming edges (root nodes). Default is True.
    return_num_in_edges : bool, optional
        If True, return the number of incoming edges for each node. Default is False.

    Returns
    -------
    torch.Tensor
        Indices of nodes, optionally sorted by number of incoming edges
    """
    incoming_edges = A.sum(dim=0)

    if exclude_root_nodes:
        nodes = torch.where(incoming_edges != 0)[0]
    else:
        nodes = torch.arange(A.shape[0])

    if sort:
        # Sort nodes by number of incoming edges in descending order
        nodes = nodes[torch.argsort(incoming_edges[nodes], descending=True)]

    if return_num_in_edges:
        return nodes, incoming_edges[nodes]
    else:
        return nodes


def get_leaf_node(A: torch.Tensor, return_num_in_edges: bool = False) -> torch.Tensor:
    """
    Get the node with the highest number of incoming edges.

    Parameters
    ----------
    A : torch.Tensor
        The adjacency matrix of shape (n, n) where A[i,j] indicates an edge from i to j
    return_num_in_edges : bool, optional
        If True, return the number of incoming edges for the node. Default is False.

    Returns
    -------
    torch.Tensor or tuple
        If return_num_in_edges is False, returns the index of the node with highest in-degree.
        If return_num_in_edges is True, returns a tuple of (node_index, num_in_edges).
    """
    result = get_nodes_by_in_degree(
        A, exclude_root_nodes=True, return_num_in_edges=return_num_in_edges, sort=True
    )
    if return_num_in_edges:
        return result[0][0], result[1][0]
    else:
        return result[0]


def calculate_all_adjacency_matrices(
    num_nodes: int, confounders=False
) -> list[torch.Tensor]:
    """
    Calculate all possible adjacency matrices for a given number of nodes that represent DAGs
    with properties that there is a single leaf node 0
    and each node at any index i has no incoming edges from nodes with index < i.

    Parameters
    ----------
    num_nodes : int
        The number of nodes in the graph
    confounders : bool, optional
        Whether to return confounded adjacency matrices. Default is False.

    Warnings
    --------
    This function generates all possible adjacency matrices for a given number of nodes,
    which can be extremely costly for large values of num_nodes. The number of matrices
    grows factorially with num_nodes, potentially leading to memory issues and long
    computation times.

    Returns
    -------
    list[torch.Tensor]
        A list of adjacency matrices of shape (num_nodes, num_nodes)
    """
    if num_nodes <= 1:
        raise ValueError(
            f"The number of nodes 'num_nodes' must be greater than 1, got {num_nodes}!"
        )
    if num_nodes == 2:
        return torch.tensor([[[0, 0], [1, 0]]])
    elif num_nodes == 3:
        graphs = [
            torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 0]]),
            torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        ]
        if confounders:
            graphs.append(torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]]))
    else:
        raise NotImplementedError(
            f"Calculation of all adjacency matrices for num_nodes > 3 is not implemented yet."
        )
    return graphs


def create_time_lag_adj_mat(
    adjacency_matrix: torch.Tensor, lag_edge_probability: float = 0.1
) -> torch.Tensor:
    """
    Create a time-lagged adjacency matrix from a static adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : torch.Tensor
        The original adjacency matrix
    lag_edge_probability : float, optional
        Probability of creating edges between time steps, default is 0.1

    Returns
    -------
    torch.Tensor
        A time-lagged adjacency matrix of shape (2, num_nodes, num_nodes)
    """
    with torch.no_grad():
        time_lagged_matrix = torch.zeros(2, *adjacency_matrix.shape)

        # Create random lagged connections based on probability
        random_mask = torch.rand_like(adjacency_matrix.float()) < lag_edge_probability
        lagged_connections = random_mask.float()

        # Set the current-impact block (same time step connections)
        time_lagged_matrix[0] = adjacency_matrix

        # Set the past-impact block (connections between time steps)
        time_lagged_matrix[1] = lagged_connections

        return time_lagged_matrix


def flatten_time_lag_adj_mat(A: torch.Tensor) -> torch.Tensor:
    """
    Flatten a time-lagged adjacency matrix.

    Parameters
    ----------
    A : torch.Tensor
        A time-lagged adjacency matrix of shape (2, num_nodes, num_nodes)

    Returns
    -------
    torch.Tensor
        A flattened adjacency matrix of shape (2*num_nodes, num_nodes)
    """
    with torch.no_grad():
        num_nodes = A.shape[1]
        A_flat = A.view(2 * num_nodes, num_nodes).contiguous()
        return A_flat


def unflatten_time_lag_adj_mat(A_flat: torch.Tensor) -> torch.Tensor:
    """
    Unflatten a flattened time-lagged adjacency matrix.

    Parameters
    ----------
    A_flat : torch.Tensor
        A flattened adjacency matrix of shape (2*num_nodes, num_nodes)

    Returns
    -------
    torch.Tensor
        A time-lagged adjacency matrix of shape (2, num_nodes, num_nodes)
    """
    with torch.no_grad():
        num_nodes = A_flat.shape[1]
        A = A_flat.view(2, num_nodes, num_nodes).contiguous()
        return A

def add_root_self_edges(A: torch.Tensor) -> torch.Tensor:
    """
    Add a self-edge to the temporal root nodes in the adjacency matrix.
    
    Root nodes are time-dependent functions driving the system.
    Thus, they should have a self-edge to themselves in the summarized graph.

    Parameters
    ----------
    A : torch.Tensor
        Adjacency matrix of shape (num_nodes, num_nodes) where A[i,j] = 1
        indicates an edge from node i to node j.

    Returns
    -------
    torch.Tensor
        Modified adjacency matrix with self-edges added to root nodes.
    """
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)
    _A = A.clone()
    root_nodes = get_root_nodes_mask(_A)
    _A[root_nodes, root_nodes] = 1
    return _A
