"""Universa: structure-switching architecture research scaffold."""

from universa.generators import (
    SwitchInstance,
    incidence_matrix,
    make_switch_instance,
    quotient_chain_map,
    random_connected_graph,
    subseed,
)
from universa.nerve import FiniteCategory, nerve_chain_complex
from universa.operators import (
    SubspaceCertificate,
    misfit,
    nullspace_basis,
    projector,
    transport_and_project,
)
from universa.structures import ChainComplex, ChainMap

__all__ = [
    "ChainComplex",
    "ChainMap",
    "FiniteCategory",
    "SubspaceCertificate",
    "SwitchInstance",
    "incidence_matrix",
    "make_switch_instance",
    "misfit",
    "nerve_chain_complex",
    "nullspace_basis",
    "projector",
    "quotient_chain_map",
    "random_connected_graph",
    "subseed",
    "transport_and_project",
]
