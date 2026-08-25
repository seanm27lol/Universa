import numpy as np
import pytest

from universa.nerve import FiniteCategory, nerve_chain_complex


def walking_category() -> FiniteCategory:
    """0 -> 1 -> 2 with identities and the forced composite."""
    morphisms = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
    compose = {
        (0, 0): 0,
        (3, 0): 3,
        (1, 1): 1,
        (4, 1): 4,
        (2, 2): 2,
        (1, 3): 3,
        (2, 4): 4,
        (4, 3): 5,
        (2, 5): 5,
        (5, 0): 5,
    }
    return FiniteCategory(3, morphisms, compose)


def group_z2_as_category() -> FiniteCategory:
    """Z/2 as a one-object category: the nerve is the bar construction."""
    morphisms = ((0, 0), (0, 0))  # identity e, generator g
    compose = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    return FiniteCategory(1, morphisms, compose)


def test_nerve_is_a_valid_complex_with_expected_dims():
    category = walking_category()
    complex_ = nerve_chain_complex(category)
    # N0 = 3 objects, N1 = 6 morphisms, N2 = 10 composable pairs.
    assert complex_.dims == (3, 6, 10)
    b1, b2 = complex_.boundaries
    assert np.abs(b1 @ b2).max() == 0.0


def test_nerve_boundary_faces():
    category = walking_category()
    b1, b2 = nerve_chain_complex(category).boundaries
    # The composable pair (f, g) = (3, 4) with composite 5 has column
    # g - gf + f in d2; find it and check.
    pairs = [
        (f, g)
        for g in range(6)
        for f in range(6)
        if category.morphisms[f][1] == category.morphisms[g][0]
    ]
    column = b2[:, pairs.index((3, 4))]
    assert column[4] == 1.0 and column[5] == -1.0 and column[3] == 1.0


def test_group_category_nerve():
    complex_ = nerve_chain_complex(group_z2_as_category())
    assert complex_.dims == (1, 2, 4)
    # One object => d1 = 0; every pair is composable => N2 = 4.
    assert np.abs(complex_.boundaries[0]).max() == 0.0


def test_invalid_category_rejected():
    with pytest.raises(ValueError, match="identity"):
        FiniteCategory(2, ((0, 1),), {})  # no identities at all
    morphisms = ((0, 0), (0, 1), (1, 1))
    with pytest.raises(ValueError, match="missing composite"):
        FiniteCategory(2, morphisms, {(0, 0): 0, (2, 2): 2})
