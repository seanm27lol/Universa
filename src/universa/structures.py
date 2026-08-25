"""The one tensor format: chain complexes and chain maps.

Every structure Universa handles compiles to a :class:`ChainComplex`: a
sequence of boundary matrices ``boundaries[k]`` mapping ``C_{k+1} -> C_k``
with ``d^2 = 0`` at every degree. Graphs are 1-complexes, cellular complexes
are themselves, sheaves add per-stalk structure on the same skeleton, and
categories arrive through :mod:`universa.nerve`.

The format is fail-closed: constructing a complex whose boundaries do not
compose to zero is an error, never a warning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

D2_TOL = 1e-12
"""Tolerance for the fail-closed ``d^2 = 0`` validation.

Integer-built complexes (graphs, nerves, quotients) compose to exact zeros;
only SVD-derived boundaries may leave numerical residue, well below this.
"""


@dataclass(frozen=True)
class ChainComplex:
    """A finite chain complex.

    ``boundaries[k]`` has shape ``(dims[k], dims[k+1])`` and maps
    ``C_{k+1} -> C_k``. A 1-complex (a graph) has exactly one boundary,
    ``B1`` of shape ``(V, E)``.
    """

    boundaries: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not self.boundaries:
            raise ValueError("a chain complex needs at least one boundary")
        for k, boundary in enumerate(self.boundaries):
            if not isinstance(boundary, np.ndarray) or boundary.ndim != 2:
                raise ValueError(f"boundaries[{k}] is not a 2-D array")
            object.__setattr__(self, "boundaries", self.boundaries)
        for k in range(len(self.boundaries) - 1):
            lower, upper = self.boundaries[k], self.boundaries[k + 1]
            if lower.shape[1] != upper.shape[0]:
                raise ValueError(
                    f"dimension mismatch at degree {k}: "
                    f"boundaries[{k}] is {lower.shape}, "
                    f"boundaries[{k + 1}] is {upper.shape}"
                )
            residue = lower @ upper
            if float(np.abs(residue).max()) > D2_TOL:
                raise ValueError(
                    f"d^2 != 0 at degree {k}: max |d d| = "
                    f"{float(np.abs(residue).max()):.3e} > {D2_TOL}"
                )

    @property
    def dims(self) -> tuple[int, ...]:
        """Dimensions of C_0 .. C_n."""
        sizes = [self.boundaries[0].shape[0]]
        sizes.extend(b.shape[1] for b in self.boundaries)
        return tuple(sizes)

    @property
    def top_degree(self) -> int:
        return len(self.boundaries)


@dataclass(frozen=True)
class ChainMap:
    """A chain map ``f: C -> D``: matrices commuting with the boundaries.

    ``maps[k]`` has shape ``(D.dims[k], C.dims[k])`` and must satisfy
    ``d_D maps[k] = maps[k-1] d_C`` at every degree ``k >= 1``. Degree 0 is
    unconstrained by commutation (there is no ``d_C`` into it).
    """

    source: ChainComplex
    target: ChainComplex
    maps: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.maps) != self.source.top_degree + 1:
            raise ValueError(
                f"expected {self.source.top_degree + 1} component maps, "
                f"got {len(self.maps)}"
            )
        if self.source.top_degree != self.target.top_degree:
            raise ValueError("source and target must share a top degree")
        for k, matrix in enumerate(self.maps):
            expected = (self.target.dims[k], self.source.dims[k])
            if matrix.shape != expected:
                raise ValueError(
                    f"maps[{k}] has shape {matrix.shape}, expected {expected}"
                )

    def commutation_residuals(self, tol: float = D2_TOL) -> tuple[float, ...]:
        """``||d_D f_k - f_{k-1} d_C||_F`` per degree ``k >= 1``.

        Exact zero for a true chain map; this is the structure-fit residual
        used for routing and discovery decisions.
        """
        residuals = []
        for k in range(1, len(self.maps)):
            left = self.target.boundaries[k - 1] @ self.maps[k]
            right = self.maps[k - 1] @ self.source.boundaries[k - 1]
            residuals.append(float(np.linalg.norm(left - right)))
        return tuple(residuals)

    def is_chain_map(self, tol: float = D2_TOL) -> bool:
        return all(r <= tol for r in self.commutation_residuals())
