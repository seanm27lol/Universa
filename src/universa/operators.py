"""The atomic move: transport along a map, project onto the consistent
subspace, and measure what had to be removed.

Everything here is exact linear algebra over float64, with certificate
residuals returned alongside every basis so callers can assert — never
assume — membership and orthonormality. This mirrors the HOMYMOLY v2
discipline: tolerances are explicit, violations are errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CERT_TOL = 1e-10
"""Maximum certified residual for nullspace membership and orthonormality."""


@dataclass(frozen=True)
class SubspaceCertificate:
    basis: np.ndarray  # orthonormal columns spanning the subspace
    membership_residual: float  # ||B Q||_F against the defining operator
    orthonormality_residual: float  # ||Q.T Q - I||_F
    observed_rank: int  # numerical rank of the defining operator


def nullspace_basis(
    operator: np.ndarray, rank_tol: float | None = None
) -> SubspaceCertificate:
    """Orthonormal basis for ``ker(operator)`` with a certificate.

    Raises if the certificate residuals exceed ``CERT_TOL`` — a basis that
    cannot be certified is a design failure, not a result.
    """
    if operator.ndim != 2:
        raise ValueError("operator must be a matrix")
    _, singulars, vh = np.linalg.svd(operator, full_matrices=True)
    if rank_tol is None:
        rank_tol = max(operator.shape) * np.finfo(float).eps * (
            singulars[0] if singulars.size else 0.0
        )
    observed_rank = int((singulars > rank_tol).sum())
    basis = vh[observed_rank:].T
    rows = operator.shape[1]
    if basis.shape != (rows, rows - observed_rank):
        raise ValueError("nullspace basis has unexpected shape")
    membership = float(np.linalg.norm(operator @ basis))
    orthonormality = float(
        np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]))
    )
    if membership > CERT_TOL or orthonormality > CERT_TOL:
        raise ValueError(
            f"nullspace certificate failed: membership {membership:.3e}, "
            f"orthonormality {orthonormality:.3e} (tol {CERT_TOL})"
        )
    return SubspaceCertificate(basis, membership, orthonormality, observed_rank)


def projector(certificate: SubspaceCertificate) -> np.ndarray:
    """The Euclidean orthogonal projector onto the certified subspace."""
    basis = certificate.basis
    proj = basis @ basis.T
    if not np.allclose(proj, proj.T, atol=CERT_TOL):
        raise ValueError("projector is not symmetric")
    idempotence = float(np.linalg.norm(proj @ proj - proj))
    if idempotence > CERT_TOL:
        raise ValueError(f"projector not idempotent: {idempotence:.3e}")
    return proj


def misfit(matrix: np.ndarray, boundary: np.ndarray) -> float:
    """``||(I - P_S) matrix||_F`` for ``S = ker(boundary)``.

    The component of ``matrix``'s columns that the structure must remove —
    the spectrally unweighted version of ``||boundary matrix||_F`` and the
    quantity HOMYMOLY's Pythagoras identity shows is exactly removable.
    """
    cert = nullspace_basis(boundary)
    proj = projector(cert)
    removed = matrix - proj @ matrix
    return float(np.linalg.norm(removed))


def transport_and_project(
    values: np.ndarray, chain_map_matrix: np.ndarray, boundary: np.ndarray
) -> tuple[np.ndarray, float]:
    """Transport ``values`` along a map and project into ``ker(boundary)``.

    Returns the projected transported values and the misfit (norm of the
    removed component). The removed component is information the source
    structure carried that the target structure forbids — the quantity a
    switching router watches when judging whether a target structure fits.
    """
    transported = chain_map_matrix @ values
    cert = nullspace_basis(boundary)
    proj = projector(cert)
    projected = proj @ transported
    removed = float(np.linalg.norm(transported - projected))
    return projected, removed
