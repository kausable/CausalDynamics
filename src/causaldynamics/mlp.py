import torch

from causaldynamics.initialization import initialize_x


def propagate_mlp_at_node(x, idx, A, W, b):
    """
    Perform a single propagation step for a node in the MLP.

    Parameters
    ----------
    x : torch.Tensor
        Current state tensor of shape (num_timesteps, num_nodes, node_dim).
    idx : int
        Index of the node to update.
    A : torch.Tensor
        Transposed adjacency matrix of shape (num_nodes, num_nodes)
        or (2*num_nodes, num_nodes) in case of time lag edges.
    W : torch.Tensor
        Weight tensor of shape (num_nodes, node_dim, node_dim).
    b : torch.Tensor
        Bias tensor of shape (num_nodes, node_dim).

    Returns
    -------
    torch.Tensor or None
        Updated values for the node at index idx, or None if the node has no incoming edges.
    """

    if A.shape[0] != A.shape[1]:  # case of time lag edges
        idx = idx % A.shape[0]

    W_idx = A[idx].bool()
    if W_idx.any():
        W_sel = W[W_idx]
        b_sel = b[W_idx]
        x_sel = x[:, W_idx]
        result = torch.matmul(W_sel, x_sel.unsqueeze(-1)).squeeze(-1) + b_sel

        return result.sum(dim=1).detach()
    else:
        return None


def propagate_mlp(A, W, b, init, standardize=False, device=None):
    """
    Propagate initial values through a directed acyclic graph using multi-layer perceptron (MLP).

    Parameters
    ----------
    A : torch.Tensor
        Adjacency matrix of shape (num_nodes, num_nodes) or (2*num_nodes, num_nodes)
        in case of time lag edges. A[i, j] = 1 indicates an edge from node i to node j.
    W : torch.Tensor
        Weight tensor of shape (num_nodes, node_dim, node_dim).
        W[i] contains the weight matrix for node i in the MLP.
    b : torch.Tensor
        Bias tensor of shape (num_nodes, node_dim) containing bias terms for each node in the MLP.
    init : torch.Tensor
        Initial values tensor of shape (num_timesteps, num_nodes, node_dim).
    standardize : bool, optional
        Whether to standardize the output of the MLP after each node propagation.
        using the internally standardized SCM (iSCM) approach. Default is False.
    device : torch.device or None, optional
        Device to place the output tensor on. Default is None.

    Returns
    -------
    torch.Tensor
        Propagated values tensor of shape (num_timesteps, num_nodes, node_dim).

    Notes
    -----
    This function processes the graph in topological order (from last node to first),
    assuming that each node at index `idx` only has incoming edges from nodes with
    indices less than `idx`. For each node, it computes a weighted sum of its inputs using
    MLP-style transformations and updates the node's value.
    """
    _, n, _ = init.shape
    x = initialize_x(init, A, device=device)
    A = A.T

    with torch.no_grad():
        for idx in reversed(range(n)):
            res = propagate_mlp_at_node(x, idx, A, W, b)

            if res is not None:
                if standardize:
                    # Standardize over the time dimension
                    res_mean = res.mean(dim=0)
                    res_var = res.var(dim=0)
                    res = (res - res_mean) / torch.sqrt(res_var)
                x[:, idx] += res

    return x.to(device)


def calculate_magnitudes(W):
    """
    Calculate the magnitudes of the weights of an MLP.

    This function computes the Frobenius norm of each weight matrix in a collection
    of MLP weight matrices. The magnitude provides a measure of the overall strength
    of the connections represented by each weight matrix.

    Parameters
    ----------
    W : torch.Tensor
        A tensor of weight matrices, typically of shape (num_nodes, input_dim, output_dim)
        where each slice W[i] represents the weight matrix for a specific node.

    Returns
    -------
    torch.Tensor
        A tensor of shape (num_nodes,) containing the magnitude (Frobenius norm)
        of each weight matrix.

    Notes
    -----
    This is a very rough proxy measure because it only computes the norm of the weight matrix
    without considering biases or the specific structure of the network. For more accurate
    measures of influence, consider methods that account for the full network dynamics.
    """
    # TODO: This is a rough proxy measure How to incorporate the biases?
    return torch.norm(W, dim=[1, 2])
