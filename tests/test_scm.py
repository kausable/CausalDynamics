from collections import Counter

import pytest
import torch

from causaldynamics.scm import (
    GNR,
    GrowingNetworkWithRedirection,
    calculate_all_adjacency_matrices,
    get_leaf_node,
    get_nodes_by_in_degree,
    get_root_nodes,
    get_root_nodes_mask,
)


def test_gnr():
    torch.manual_seed(42)

    n = [10, 100, 1000]
    r = [0.0, 0.1, 1.0]

    for n, r in zip(n, r):
        gnr = GNR(num_nodes=n, redirect_prob=r)
        G = gnr.generate()
        assert G.shape == (n, n)

        # Test that the first element is the largest. Test with nodes after the fifth
        # to ensure that statistically this test passes
        assert all(gnr.A_k[0] > gnr.A_k[i] for i in range(5, len(gnr.A_k)))

        # Check that most nodes have 1 incoming link
        assert Counter(gnr.A_k.tolist()).most_common(1)[0][0] == 1

        # Assert that the sum of the rows of G plus one
        # (k-1 incoming links and 1 outgoing link) is equal to A_k
        assert (G.sum(axis=0) + 1 == gnr.A_k).all()

        with pytest.raises(
            ValueError,
            match="The probability 'redirect_prob' must be between 0. and 1, got -0.1!",
        ):
            GNR(10, -0.1).generate()

        with pytest.raises(
            ValueError,
            match="The probability 'redirect_prob' must be between 0. and 1, got 1.1!",
        ):
            GNR(10, 1.1).generate()

        with pytest.raises(
            ValueError,
            match="The number of nodes 'num_nodes' must be greater than 0, got -10!",
        ):
            GNR(-10, 0.5).generate()


def test_gnr_alias():
    """Test that GNR is an alias for GrowingNetworkWithRedirection."""
    assert GNR is GrowingNetworkWithRedirection


def test_get_root_nodes_mask():
    """Test that get_root_nodes_mask returns the correct mask."""
    A = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    assert torch.allclose(
        get_root_nodes_mask(A), torch.tensor([False, False, True, True])
    )
    A = torch.tensor([[0, 0], [1, 0]])
    assert torch.allclose(get_root_nodes_mask(A), torch.tensor([False, True]))


def test_get_root_nodes():
    """Test that get_root_nodes returns the correct nodes."""
    A = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    assert torch.allclose(get_root_nodes(A), torch.tensor([2, 3]))


def test_get_nodes_by_incoming_degree():
    """Test that get_nodes_by_in_degree returns the correct nodes."""
    A = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])

    # Test default behavior
    assert torch.allclose(get_nodes_by_in_degree(A), torch.tensor([1, 0]))

    # Test with return_num_in_edges=True
    nodes, in_edges = get_nodes_by_in_degree(A, return_num_in_edges=True)
    assert torch.allclose(nodes, torch.tensor([1, 0]))
    assert torch.allclose(in_edges, torch.tensor([2, 1]))

    # Test with sort=False
    assert torch.allclose(get_nodes_by_in_degree(A, sort=False), torch.tensor([0, 1]))

    # Test with exclude_root_nodes=False
    assert torch.allclose(
        get_nodes_by_in_degree(A, exclude_root_nodes=False), torch.tensor([1, 0, 2, 3])
    )

    # Test with both sort=False and exclude_root_nodes=False
    assert torch.allclose(
        get_nodes_by_in_degree(A, sort=False, exclude_root_nodes=False),
        torch.tensor([0, 1, 2, 3]),
    )

    # Test with a more complex graph
    A = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]])

    # Test default behavior
    assert torch.allclose(get_nodes_by_in_degree(A), torch.tensor([0, 1, 2]))

    # Test with return_num_in_edges=True
    nodes, in_edges = get_nodes_by_in_degree(A, return_num_in_edges=True)
    assert torch.allclose(nodes, torch.tensor([0, 1, 2]))
    assert torch.allclose(in_edges, torch.tensor([3, 2, 1]))


def test_get_leaf_node():
    # Test with a simple graph
    A = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])

    # Test default behavior
    assert get_leaf_node(A) == 1

    # Test with return_num_in_edges=True
    leaf, num_edges = get_leaf_node(A, return_num_in_edges=True)
    assert leaf == 1
    assert num_edges == 2

    # Test with a more complex graph
    A = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]])

    # Test default behavior
    assert get_leaf_node(A) == 0

    # Test with return_num_in_edges=True
    leaf, num_edges = get_leaf_node(A, return_num_in_edges=True)
    assert leaf == 0
    assert num_edges == 3


def test_calculate_all_adjacency_matrices():
    """Test that calculate_all_adjacency_matrices returns the correct adjacency matrices."""
    A_N2 = torch.tensor([[0, 0], [1, 0]])

    assert torch.allclose(calculate_all_adjacency_matrices(num_nodes=2), A_N2)

    A_N3 = [
        torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 0]]),
        torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
    ]

    A_N3_conf = [
        torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 0]]),
        torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0]]),
    ]

    for i, A in enumerate(calculate_all_adjacency_matrices(num_nodes=3)):
        assert torch.allclose(A, A_N3[i])

    for i, A in enumerate(
        calculate_all_adjacency_matrices(num_nodes=3, confounders=True)
    ):
        assert torch.allclose(A, A_N3_conf[i])
