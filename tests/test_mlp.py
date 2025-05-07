import numpy as np
import torch

from causaldynamics.mlp import calculate_magnitudes, propagate_mlp


def test_propagate_chain_of_length_3():
    """
    Test the propagation of a chain of length 3.

    Graph: 0<-1<-2
    """
    L = 1
    N = 3
    d = 2
    A = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    W = torch.ones(N, d, d)
    for i in range(N):
        W[i] = W[i] * 0.1 * i + 0.1
    b = torch.tensor([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])

    init = torch.ones(L, N, d)
    data = propagate_mlp(A, W, b, init)

    # Nodes without incoming edges are not changed
    x20 = 1.0
    x21 = 1.0
    assert torch.allclose(data[:, 2], torch.tensor([[x20, x21]]))

    # Check that node 1 is calculated correctly
    x10 = W[2, 0, 0] * x20 + W[2, 0, 1] * x21 + b[2, 0]
    x11 = W[2, 1, 0] * x21 + W[2, 1, 1] * x20 + b[2, 1]
    assert torch.allclose(data[:, 1], torch.tensor([[x10, x11]]))

    # Check that node 2 is calculated correctly
    x00 = W[1, 0, 0] * x10 + W[1, 0, 1] * x11 + b[1, 0]
    x01 = W[1, 1, 0] * x11 + W[1, 1, 1] * x10 + b[1, 1]
    assert torch.allclose(data[:, 0], torch.tensor([[x00, x01]]))


def test_propagate_small_scm():
    """
    Test the propagation of a small SCM.

    Graph:  0<-1<-2
            0<-3
    """
    L = 1
    N = 4
    d = 2
    A = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    W = torch.ones(N, d, d)
    for i in range(N):
        W[i] = W[i] * 0.1 * i + 0.1
    b = torch.tensor([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08]])

    init = torch.ones(L, N, d)
    data = propagate_mlp(A, W, b, init)

    # Nodes without incoming edge are not changed
    x30 = 1.0
    x31 = 1.0
    x20 = 1.0
    x21 = 1.0
    assert torch.allclose(data[:, 3], torch.tensor([[x30, x31]]))
    assert torch.allclose(data[:, 2], torch.tensor([[x20, x21]]))

    # Node 1 that has 1 incoming edge from node 2 is calculated correctly
    x10 = W[2, 0, 0] * x20 + W[2, 0, 1] * x21 + b[2, 0]
    x11 = W[2, 1, 0] * x21 + W[2, 1, 1] * x20 + b[2, 1]
    assert torch.allclose(data[:, 1], torch.tensor([[x10, x11]]))

    # Node 0 that has 2 incoming edges from node 1 and node 3 is calculated correctly
    x00 = (W[3, 0, 0] * x30 + W[3, 0, 1] * x31 + b[3, 0]) + (
        W[1, 0, 0] * x10 + W[1, 0, 1] * x11 + b[1, 0]
    )
    x01 = (W[3, 1, 0] * x31 + W[3, 1, 1] * x30 + b[3, 1]) + (
        W[1, 1, 0] * x11 + W[1, 1, 1] * x10 + b[1, 1]
    )
    assert torch.allclose(data[:, 0], torch.tensor([[x00, x01]]))



def test_propagate_with_confounder():
    """
    Test the propagation with confounders.

    Graph:  0<-1<-3
            0<-2<-3
    """
    L = 1
    N = 4
    d = 2
    A = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]])
    W = torch.ones(N, d, d)
    for i in range(N):
        W[i] = W[i] * 0.1 * i + 0.1
    b = torch.tensor([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08]])

    init = torch.ones(L, N, d)
    data = propagate_mlp(A, W, b, init)

    # Node 3 without incoming edges are not changed
    x30 = 1.0
    x31 = 1.0
    assert torch.allclose(data[:, 3], torch.tensor([[x30, x31]]))

    # Node 2 that has 1 incoming edges from node 3
    x20 = W[3, 0, 0] * x30 + W[3, 0, 1] * x31 + b[3, 0]
    x21 = W[3, 1, 0] * x31 + W[3, 1, 1] * x30 + b[3, 1]
    assert torch.allclose(data[:, 2], torch.tensor([[x20, x21]]))

    # Node 1 that has 1 incoming edges from node 3
    x10 = W[3, 0, 0] * x30 + W[3, 0, 1] * x31 + b[3, 0]
    x11 = W[3, 1, 0] * x31 + W[3, 1, 1] * x30 + b[3, 1]
    assert torch.allclose(data[:, 1], torch.tensor([[x10, x11]]))

    # Node 0 that has 1 incoming edges from node 1 and one from node 2
    x00 = (
        W[1, 0, 0] * x10
        + W[1, 0, 1] * x11
        + b[1, 0]
        + W[2, 0, 0] * x20
        + W[2, 0, 1] * x21
        + b[2, 0]
    )
    x01 = (
        W[1, 1, 0] * x11
        + W[1, 1, 1] * x10
        + b[1, 1]
        + W[2, 1, 0] * x21
        + W[2, 1, 1] * x20
        + b[2, 1]
    )
    assert torch.allclose(data[:, 0], torch.tensor([[x00, x01]]))


def test_calculate_magnitudes():
    """
    Test the calculation of the magnitudes of the weights and biases of an MLP.
    """
    num_nodes = 3
    W = torch.ones(num_nodes, 2, 2)
    for i in range(num_nodes):
        for j in range(2):
            for k in range(2):
                W[i, j, k] = W[i, j, k] * 0.1 * i + 1.0 * j + 10.0 * k

    magnitudes = calculate_magnitudes(W)

    # Check that magnitudes is 1-dimensional
    assert len(magnitudes.shape) == 1
    assert magnitudes.shape[0] == num_nodes

    # Calculate expected magnitudes manually for verification
    expected_magnitudes = torch.zeros(num_nodes)
    for i in range(num_nodes):
        expected_magnitudes[i] = torch.norm(W[i])

    # Verify the magnitudes match our expectations
    assert torch.allclose(magnitudes, expected_magnitudes)
