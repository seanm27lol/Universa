"""Cellular sheaves on graphs: the second synthetic family.

A cellular sheaf on a graph attaches a vector space (stalk) to every vertex
and every edge, plus a restriction map ``rho_{v,e}: R^{d_v} -> R^{d_e}`` for
every incidence. Its cochain spaces are the direct sums of stalks,
``C^0 = sum_v R^{d_v}`` and ``C^1 = sum_e R^{d_e}``, and its coboundary is
the block matrix

    (delta x)_e = rho_{h,e} x_h - rho_{t,e} x_t     for e = (t -> h),

with signs read off the base B1. The harmonic space ``ker(delta) = H^0`` is
the sheaf's consistent subspace — the object the atomic move projects onto.

Re-indexing into the one tensor format: cochains become chains by
``C_0 := C^1`` (edge stalks) and ``C_1 := C^0`` (vertex stalks), with the
coboundary ``delta`` itself as the single boundary. Then ``H^0`` is
literally the cycle space of the compiled complex and
:mod:`universa.operators` applies verbatim. A sheaf morphism (stalk maps
``phi_v``, ``phi_e`` with ``rho'_{v,e} phi_v = phi_e rho_{v,e}`` per
incidence) compiles to a chain map with components
``(diag(phi_e), diag(phi_v))``; its single commutation residual is the
root-sum-square of the per-incidence naturality residuals, so the ChainMap
machinery audits sheaf morphisms exactly.

Exactness: planted morphisms build the target *from* the source by
``rho' := phi_e rho phi_v^{-1}`` with each ``phi`` a signed permutation
matrix scaled by +/-1 or +/-2. Every entry of ``rho'`` is then a permuted
source entry times a power of two — an exponent shift, never mantissa
rounding — so naturality residuals are exact zeros in float64, and no
arithmetic beyond dyadic fractions ever arises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from universa.generators import (
    SwitchInstance,
    random_connected_graph,
    subseed,
)
from universa.structures import D2_TOL, ChainComplex, ChainMap

MORPHISM_SCALES = (-2, -1, 1, 2)
"""Power-of-two scales for planted stalk isomorphisms: dyadic-exact and
perfectly conditioned (a scaled permutation has condition number 1)."""


def _edge_endpoints(boundary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(tail, head) per edge from B1's sign pattern; fail-closed on non-graphs."""
    tails = np.zeros(boundary.shape[1], dtype=int)
    heads = np.zeros(boundary.shape[1], dtype=int)
    for e in range(boundary.shape[1]):
        column = boundary[:, e]
        tail_candidates = np.flatnonzero(column == -1.0)
        head_candidates = np.flatnonzero(column == 1.0)
        if (
            len(tail_candidates) != 1
            or len(head_candidates) != 1
            or np.count_nonzero(column) != 2
        ):
            raise ValueError(
                f"column {e} is not a graph incidence column "
                "(exactly one -1 and one +1)"
            )
        tails[e] = tail_candidates[0]
        heads[e] = head_candidates[0]
    return tails, heads


def _block_diag(blocks: tuple[np.ndarray, ...]) -> np.ndarray:
    """Dense block-diagonal assembly (numpy-only stand-in for scipy's)."""
    rows = sum(block.shape[0] for block in blocks)
    cols = sum(block.shape[1] for block in blocks)
    out = np.zeros((rows, cols))
    r = c = 0
    for block in blocks:
        out[r : r + block.shape[0], c : c + block.shape[1]] = block
        r += block.shape[0]
        c += block.shape[1]
    return out


@dataclass(frozen=True)
class Sheaf:
    """A cellular sheaf on a graph (a 1-complex).

    ``restrictions[(v, e)]`` is the dense float64 block
    ``rho_{v,e}: R^{d_v} -> R^{d_e}`` of shape
    ``(edge_dims[e], vertex_dims[v])``. Every edge must carry exactly its
    two endpoint restrictions, no more; the base must be a genuine graph
    incidence matrix.
    """

    base: ChainComplex
    vertex_dims: tuple[int, ...]
    edge_dims: tuple[int, ...]
    restrictions: dict[tuple[int, int], np.ndarray]

    def __post_init__(self) -> None:
        if self.base.top_degree != 1:
            raise ValueError("a sheaf lives on a graph (a 1-complex)")
        boundary = self.base.boundaries[0]
        num_vertices, num_edges = boundary.shape
        tails, heads = _edge_endpoints(boundary)
        for d in (*self.vertex_dims, *self.edge_dims):
            if not isinstance(d, (int, np.integer)):
                raise ValueError("stalk dimensions must be integers")
        vertex_dims = tuple(int(d) for d in self.vertex_dims)
        edge_dims = tuple(int(d) for d in self.edge_dims)
        if len(vertex_dims) != num_vertices or len(edge_dims) != num_edges:
            raise ValueError("stalk dims must cover every vertex and every edge")
        if any(d < 1 for d in (*vertex_dims, *edge_dims)):
            raise ValueError("stalk dimensions must be positive")
        restrictions: dict[tuple[int, int], np.ndarray] = {}
        for (v, e), block in self.restrictions.items():
            v, e = int(v), int(e)
            block = np.asarray(block, dtype=np.float64)
            if block.ndim != 2:
                raise ValueError(f"restriction {(v, e)} is not a 2-D block")
            if not (0 <= v < num_vertices and 0 <= e < num_edges):
                raise ValueError(f"restriction {(v, e)} is out of range")
            if block.shape != (edge_dims[e], vertex_dims[v]):
                raise ValueError(
                    f"restriction {(v, e)} has shape {block.shape}, "
                    f"expected {(edge_dims[e], vertex_dims[v])}"
                )
            if not np.isfinite(block).all():
                raise ValueError(f"restriction {(v, e)} has non-finite entries")
            restrictions[(v, e)] = block
        expected = {
            (int(v), e) for e in range(num_edges) for v in (tails[e], heads[e])
        }
        if set(restrictions) != expected:
            missing = expected - set(restrictions)
            extra = set(restrictions) - expected
            raise ValueError(
                "restrictions must cover exactly the incidences "
                f"(missing {sorted(missing)}, extra {sorted(extra)})"
            )
        object.__setattr__(self, "vertex_dims", vertex_dims)
        object.__setattr__(self, "edge_dims", edge_dims)
        object.__setattr__(self, "restrictions", restrictions)

    @property
    def num_vertices(self) -> int:
        return self.base.boundaries[0].shape[0]

    @property
    def num_edges(self) -> int:
        return self.base.boundaries[0].shape[1]

    @property
    def c0_dim(self) -> int:
        """Dimension of C^0 (the direct sum of vertex stalks)."""
        return sum(self.vertex_dims)

    @property
    def c1_dim(self) -> int:
        """Dimension of C^1 (the direct sum of edge stalks)."""
        return sum(self.edge_dims)

    @property
    def incidence_pairs(self) -> tuple[tuple[int, int], ...]:
        """Every incidence ``(v, e)``, ordered by edge, tail before head."""
        tails, heads = _edge_endpoints(self.base.boundaries[0])
        return tuple(
            (int(v), e)
            for e in range(self.num_edges)
            for v in (tails[e], heads[e])
        )


def coboundary(sheaf: Sheaf) -> np.ndarray:
    """The block coboundary ``delta: C^0 -> C^1``, shape ``(c1_dim, c0_dim)``.

    ``(delta x)_e = rho_{h,e} x_h - rho_{t,e} x_t`` for ``e = (t -> h)``,
    with the tail/head signs taken from the base B1.
    """
    tails, heads = _edge_endpoints(sheaf.base.boundaries[0])
    delta = np.zeros((sheaf.c1_dim, sheaf.c0_dim))
    vertex_offsets = np.concatenate(([0], np.cumsum(sheaf.vertex_dims)))
    edge_offsets = np.concatenate(([0], np.cumsum(sheaf.edge_dims)))
    for e in range(sheaf.num_edges):
        rows = slice(edge_offsets[e], edge_offsets[e + 1])
        for v, sign in ((int(tails[e]), -1.0), (int(heads[e]), 1.0)):
            cols = slice(vertex_offsets[v], vertex_offsets[v + 1])
            delta[rows, cols] = sign * sheaf.restrictions[(v, e)]
    return delta


def to_chain_complex(sheaf: Sheaf) -> ChainComplex:
    """The sheaf as a chain complex: ``C_0 := C^1``, ``C_1 := C^0``, boundary ``delta``.

    ``ker(delta) = H^0(sheaf)`` is the cycle space of this complex, so the
    certified-subspace machinery of :mod:`universa.operators` applies
    verbatim.
    """
    return ChainComplex((coboundary(sheaf),))


@dataclass(frozen=True)
class SheafMorphism:
    """Stalk maps between two sheaves on a common base graph.

    ``vertex_maps[v]`` has shape
    ``(target.vertex_dims[v], source.vertex_dims[v])`` and ``edge_maps[e]``
    likewise. Naturality per incidence is
    ``rho'_{v,e} phi_v = phi_e rho_{v,e}``; the residuals below are exact
    zeros for a true morphism.
    """

    source: Sheaf
    target: Sheaf
    vertex_maps: tuple[np.ndarray, ...]
    edge_maps: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not np.array_equal(
            self.source.base.boundaries[0], self.target.base.boundaries[0]
        ):
            raise ValueError("a sheaf morphism requires a common base graph")
        if len(self.vertex_maps) != self.source.num_vertices:
            raise ValueError("need one stalk map per vertex")
        if len(self.edge_maps) != self.source.num_edges:
            raise ValueError("need one stalk map per edge")
        vertex_maps = []
        for v, phi in enumerate(self.vertex_maps):
            phi = np.asarray(phi, dtype=np.float64)
            expected = (self.target.vertex_dims[v], self.source.vertex_dims[v])
            if phi.shape != expected:
                raise ValueError(
                    f"vertex_maps[{v}] has shape {phi.shape}, expected {expected}"
                )
            if not np.isfinite(phi).all():
                raise ValueError(f"vertex_maps[{v}] has non-finite entries")
            vertex_maps.append(phi)
        edge_maps = []
        for e, phi in enumerate(self.edge_maps):
            phi = np.asarray(phi, dtype=np.float64)
            expected = (self.target.edge_dims[e], self.source.edge_dims[e])
            if phi.shape != expected:
                raise ValueError(
                    f"edge_maps[{e}] has shape {phi.shape}, expected {expected}"
                )
            if not np.isfinite(phi).all():
                raise ValueError(f"edge_maps[{e}] has non-finite entries")
            edge_maps.append(phi)
        object.__setattr__(self, "vertex_maps", tuple(vertex_maps))
        object.__setattr__(self, "edge_maps", tuple(edge_maps))

    def naturality_residuals(self) -> tuple[float, ...]:
        """``||rho'_{v,e} phi_v - phi_e rho_{v,e}||_F`` per incidence.

        Ordered as ``Sheaf.incidence_pairs``; exact zero for a true
        morphism. This is the stalk-level structure-fit residual.
        """
        residuals = []
        for v, e in self.source.incidence_pairs:
            left = self.target.restrictions[(v, e)] @ self.vertex_maps[v]
            right = self.edge_maps[e] @ self.source.restrictions[(v, e)]
            residuals.append(float(np.linalg.norm(left - right)))
        return tuple(residuals)

    def is_morphism(self, tol: float = D2_TOL) -> bool:
        return all(r <= tol for r in self.naturality_residuals())

    def to_chain_map(self) -> ChainMap:
        """Compile to a chain map with components ``(diag(phi_e), diag(phi_v))``.

        The single commutation residual
        ``||delta' diag(phi_v) - diag(phi_e) delta||_F`` equals the
        root-sum-square of the per-incidence naturality residuals (the B1
        signs only flip block signs), so chain-map auditing and stalk-level
        auditing report the same number.
        """
        return ChainMap(
            to_chain_complex(self.source),
            to_chain_complex(self.target),
            (_block_diag(self.edge_maps), _block_diag(self.vertex_maps)),
        )


def _random_restrictions(
    seed: int,
    base: ChainComplex,
    vertex_dims: tuple[int, ...],
    edge_dims: tuple[int, ...],
) -> dict[tuple[int, int], np.ndarray]:
    """Dense integer blocks with entries in {-2..2} for every incidence."""
    rng = np.random.default_rng(seed)
    tails, heads = _edge_endpoints(base.boundaries[0])
    restrictions = {}
    for e in range(base.boundaries[0].shape[1]):
        for v in (int(tails[e]), int(heads[e])):
            block = rng.integers(-2, 3, size=(edge_dims[e], vertex_dims[v]))
            restrictions[(v, e)] = block.astype(np.float64)
    return restrictions


def random_sheaf(
    seed: int,
    num_vertices: int = 6,
    num_edges: int = 9,
    max_stalk_dim: int = 3,
) -> Sheaf:
    """A random sheaf: connected base graph, stalk dims 1..max_stalk_dim,
    restriction entries drawn from {-2..2} (exact integers)."""
    base = random_connected_graph(
        subseed(seed, "sheaf", "base"), num_vertices, num_edges
    )
    rng = np.random.default_rng(subseed(seed, "sheaf", "stalks"))
    vertex_dims = tuple(
        int(d) for d in rng.integers(1, max_stalk_dim + 1, size=num_vertices)
    )
    edge_dims = tuple(
        int(d) for d in rng.integers(1, max_stalk_dim + 1, size=num_edges)
    )
    restrictions = _random_restrictions(
        subseed(seed, "sheaf", "restrictions"), base, vertex_dims, edge_dims
    )
    return Sheaf(base, vertex_dims, edge_dims, restrictions)


def planted_morphism(source: Sheaf, seed: int) -> tuple[Sheaf, SheafMorphism]:
    """A target sheaf built BY a morphism, so naturality is exact.

    Each stalk map is a signed permutation scaled by +/-1 or +/-2, and the
    target restrictions are ``rho'_{v,e} := phi_e rho_{v,e} phi_v^{-1}``,
    computed as ``(phi_e rho phi_v^T) / scale_v**2``. Every entry of
    ``rho'`` is a permuted source entry times a power of two — an exponent
    shift with no mantissa rounding, for any float64 source entries — so
    the naturality residuals are exactly 0.0. Verified fail-closed: a
    planted morphism whose residual is not exactly zero is a design
    failure, not a result.
    """
    rng = np.random.default_rng(subseed(seed, "sheaf", "morphism"))

    def draw_stalk_map(dim: int) -> tuple[np.ndarray, int]:
        permutation = rng.permutation(dim)
        scale = MORPHISM_SCALES[int(rng.integers(len(MORPHISM_SCALES)))]
        return scale * np.eye(dim)[list(permutation)], scale

    drawn_v = [draw_stalk_map(d) for d in source.vertex_dims]
    drawn_e = [draw_stalk_map(d) for d in source.edge_dims]
    vertex_maps = tuple(phi for phi, _ in drawn_v)
    edge_maps = tuple(phi for phi, _ in drawn_e)
    vertex_scales = [scale for _, scale in drawn_v]

    restrictions = {}
    for v, e in source.incidence_pairs:
        rho = source.restrictions[(v, e)]
        # phi_v^{-1} = phi_v^T / scale_v**2 since phi_v = scale_v * permutation.
        rho_prime = (
            edge_maps[e] @ rho @ vertex_maps[v].T
        ) / vertex_scales[v] ** 2
        restrictions[(v, e)] = rho_prime
    target = Sheaf(
        source.base, source.vertex_dims, source.edge_dims, restrictions
    )
    morphism = SheafMorphism(source, target, vertex_maps, edge_maps)
    residuals = morphism.naturality_residuals()
    if any(r != 0.0 for r in residuals):
        raise RuntimeError(
            f"planted morphism is not exact: residuals {residuals}"
        )
    return target, morphism


def make_sheaf_switch_instance(
    seed: int,
    num_vertices: int = 6,
    num_edges: int = 9,
    max_stalk_dim: int = 3,
    num_decoys: int = 3,
) -> SwitchInstance:
    """A sheaf routing instance, reusing :class:`SwitchInstance` directly.

    The planted stalk maps are per-stalk isomorphisms, so source, target,
    and every candidate compile to 1-complexes with identical dims and the
    planted morphism compiles to a ChainMap shape-compatible with them all
    (a SheafSwitchInstance would only be needed for morphisms that change
    stalk dims). Decoys share the base graph and stalk dims but redraw the
    restriction maps, and are accepted only when the planted map's
    commutation residual against them exceeds 1e-9, so discrimination is
    guaranteed by construction.
    """
    source = random_sheaf(seed, num_vertices, num_edges, max_stalk_dim)
    _, morphism = planted_morphism(source, seed)
    chain_map = morphism.to_chain_map()

    decoys: list[ChainComplex] = []
    attempt = 0
    while len(decoys) < num_decoys:
        attempt += 1
        if attempt > 1000:
            raise RuntimeError("could not generate discriminating decoys")
        draw = subseed(seed, "sheaf", "decoy", str(attempt))
        restrictions = _random_restrictions(
            draw, source.base, source.vertex_dims, source.edge_dims
        )
        candidate = to_chain_complex(
            Sheaf(source.base, source.vertex_dims, source.edge_dims, restrictions)
        )
        probe = ChainMap(chain_map.source, candidate, chain_map.maps)
        (residual,) = probe.commutation_residuals()
        if residual > 1e-9:
            decoys.append(candidate)
    return SwitchInstance(
        seed, chain_map.source, chain_map.target, chain_map, tuple(decoys)
    )
