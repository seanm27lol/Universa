"""Universa: structure-switching architecture research scaffold."""

from universa.category_instances import (
    GroupSwitchInstance,
    cyclic_group,
    group_as_category,
    induced_nerve_map,
    make_group_switch_instance,
    symmetric_group_3,
)
from universa.complexes2 import (
    induced_quotient_map,
    make_two_complex_switch_instance,
    two_complex,
)
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
from universa.sheaves import (
    Sheaf,
    SheafMorphism,
    coboundary,
    make_sheaf_switch_instance,
    planted_morphism,
    random_sheaf,
    to_chain_complex,
)
from universa.structures import ChainComplex, ChainMap

__all__ = [
    "ChainComplex",
    "ChainMap",
    "FiniteCategory",
    "GroupSwitchInstance",
    "Sheaf",
    "SheafMorphism",
    "SubspaceCertificate",
    "SwitchInstance",
    "coboundary",
    "cyclic_group",
    "group_as_category",
    "incidence_matrix",
    "induced_nerve_map",
    "induced_quotient_map",
    "make_group_switch_instance",
    "make_sheaf_switch_instance",
    "make_switch_instance",
    "make_two_complex_switch_instance",
    "misfit",
    "nerve_chain_complex",
    "nullspace_basis",
    "planted_morphism",
    "projector",
    "quotient_chain_map",
    "random_connected_graph",
    "random_sheaf",
    "subseed",
    "symmetric_group_3",
    "to_chain_complex",
    "transport_and_project",
    "two_complex",
]
