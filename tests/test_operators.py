import numpy as np
import pytest

from universa.operators import (
    CERT_TOL,
    misfit,
    nullspace_basis,
    projector,
    transport_and_project,
)


def incidence() -> np.ndarray:
    # Path graph on 4 vertices: a tree, so ker(B1) = {0} (no cycles).
    boundary = np.zeros((4, 3))
    for edge in range(3):
        boundary[edge, edge] = -1.0
        boundary[edge + 1, edge] = 1.0
    return boundary


def cycle_incidence() -> np.ndarray:
    # Triangle: ker(B1) is the cycle, dimension 1.
    return np.array(
        [[-1.0, 0.0, -1.0], [1.0, -1.0, 0.0], [0.0, 1.0, 1.0]]
    )


def test_nullspace_basis_certificate_is_exact():
    cert = nullspace_basis(cycle_incidence())
    assert cert.basis.shape == (3, 1)
    assert cert.membership_residual <= CERT_TOL
    assert cert.orthonormality_residual <= CERT_TOL
    assert cert.observed_rank == 2
    # The kernel of the triangle incidence is the cycle (1, 1, -1) / sqrt(3).
    assert np.allclose(np.abs(cert.basis[:, 0]), 1.0 / np.sqrt(3.0))


def test_tree_has_trivial_cycle_space():
    cert = nullspace_basis(incidence())
    assert cert.basis.shape == (3, 0)
    assert cert.observed_rank == 3


def test_projector_properties():
    proj = projector(nullspace_basis(cycle_incidence()))
    assert np.allclose(proj, proj.T, atol=CERT_TOL)
    assert np.linalg.norm(proj @ proj - proj) <= CERT_TOL


def test_misfit_zero_inside_and_positive_outside():
    boundary = cycle_incidence()
    cert = nullspace_basis(boundary)
    inside = cert.basis @ np.array([[2.0, -1.0, 0.5]])
    assert misfit(inside, boundary) <= CERT_TOL
    outside = np.eye(3)
    assert misfit(outside, boundary) > 0.5


def test_transport_and_project_removes_off_cycle_component():
    boundary = cycle_incidence()
    cert = nullspace_basis(boundary)
    proj = projector(cert)
    rng = np.random.default_rng(7)
    on_cycle = cert.basis @ rng.standard_normal((1, 4))
    off_cycle = (np.eye(3) - proj) @ rng.standard_normal((3, 4))
    mapping = on_cycle + off_cycle
    values = rng.standard_normal((4, 5))
    projected, removed = transport_and_project(values, mapping, boundary)
    assert np.linalg.norm(boundary @ projected) <= CERT_TOL
    assert removed > 0.0
    # Pythagoras: projected part equals the on-cycle transport exactly.
    expected = on_cycle @ values
    assert np.allclose(projected, expected, atol=CERT_TOL)


def test_uncertifiable_basis_is_an_error():
    # An operator with NaN cannot produce a certified basis.
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        nullspace_basis(np.array([[np.nan, 1.0]]))
