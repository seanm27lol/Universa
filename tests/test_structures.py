import numpy as np
import pytest

from universa.structures import ChainComplex, ChainMap


def path_graph(num_vertices: int) -> ChainComplex:
    boundary = np.zeros((num_vertices, num_vertices - 1))
    for edge in range(num_vertices - 1):
        boundary[edge, edge] = -1.0
        boundary[edge + 1, edge] = 1.0
    return ChainComplex((boundary,))


def triangle() -> ChainComplex:
    b1 = np.array(
        [[-1.0, 0.0, -1.0], [1.0, -1.0, 0.0], [0.0, 1.0, 1.0]]
    )
    b2 = np.array([[1.0], [1.0], [-1.0]])
    return ChainComplex((b1, b2))


def test_dims_and_degrees():
    assert path_graph(4).dims == (4, 3)
    assert path_graph(4).top_degree == 1
    two_cell = triangle()
    assert two_cell.dims == (3, 3, 1)
    assert two_cell.top_degree == 2


def test_d_squared_is_enforced():
    b1 = np.array([[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]])  # path on 3
    bad_b2 = np.array([[1.0], [1.0]])  # B1 B2 = [-1, 0, 1]^T != 0
    with pytest.raises(ValueError, match=r"d\^2 != 0"):
        ChainComplex((b1, bad_b2))


def test_dimension_mismatch_is_enforced():
    b1 = np.array([[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]])
    wrong_width = np.zeros((4, 1))
    with pytest.raises(ValueError, match="dimension mismatch"):
        ChainComplex((b1, wrong_width))


def test_chain_map_commutation_and_residuals():
    source = path_graph(3)
    target = path_graph(2)
    # Collapse vertices 0,1 -> 0 and 2 -> 1: edge (0,1) collapses, edge
    # (1,2) maps to the single target edge with matching orientation.
    f0 = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    f1 = np.array([[0.0, 1.0]])
    chain_map = ChainMap(source, target, (f0, f1))
    assert chain_map.is_chain_map()
    (residual,) = chain_map.commutation_residuals()
    assert residual == 0.0


def test_chain_map_detects_violation():
    source = path_graph(3)
    target = path_graph(2)
    f0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    f1 = np.array([[0.0, 1.0]])  # inconsistent with f0
    probe = ChainMap(source, target, (f0, f1))
    assert not probe.is_chain_map()
    (residual,) = probe.commutation_residuals()
    assert residual > 1.0


def test_chain_map_shape_checks():
    source = path_graph(3)
    target = path_graph(2)
    with pytest.raises(ValueError, match="component maps"):
        ChainMap(source, target, (np.eye(2, 3),))
    with pytest.raises(ValueError, match="maps\\[0\\]"):
        ChainMap(source, target, (np.eye(3), np.array([[1.0, 0.0]])))
