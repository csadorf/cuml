# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest
from hypothesis import example, given, note
from hypothesis import strategies as st
from hypothesis import target
from hypothesis.extra.numpy import (
    array_shapes,
    arrays,
    floating_dtypes,
    integer_dtypes,
)

from cuml.testing.utils import (
    array_difference,
    array_equal,
    assert_array_equal,
)


@pytest.mark.parametrize(
    "dtype,min_value",
    [
        (np.int8, -128),
        (np.int16, -32768),
        (np.int32, -2147483648),
        (np.int64, -9223372036854775808),
    ],
)
@pytest.mark.parametrize("with_sign", [True, False])
def test_array_equal_integer_overflow(dtype, min_value, with_sign):
    """Test that array comparison handles integer overflow in abs() correctly.

    When taking abs() of the minimum value of a signed integer type,
    overflow occurs because the positive value cannot be represented.
    For example: np.abs(np.int8(-128)) -> -128 (overflow).

    This test ensures the comparison logic handles this edge case correctly
    by casting to float64 before taking absolute values.
    """
    # Create arrays with the minimum value that causes overflow
    a = np.array([[0, 0], [0, 0]], dtype=dtype)
    b = np.array([[min_value, 0], [1, -1]], dtype=dtype)

    # These should not raise an error and should compute differences correctly
    equal = array_equal(a, b, unit_tol=1.0, with_sign=with_sign)
    difference = array_difference(a, b, with_sign=with_sign)

    # Arrays are clearly different, so equal should be False
    assert not equal
    # Difference should be non-zero
    assert difference != 0
    # For with_sign=False, difference should be positive and finite
    if not with_sign:
        assert difference > 0
        assert np.isfinite(difference)


@example(array=np.array([1, 2, 3]), tol=1e-4)
@given(
    arrays(
        dtype=st.one_of(floating_dtypes(), integer_dtypes()),
        shape=array_shapes(),
    ),
    st.floats(1e-4, 1.0),
)
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract")
def test_array_equal_same_array(array, tol):
    equal = array_equal(array, array, tol)
    note(equal)
    difference = equal.compute_difference()
    if np.isfinite(difference):
        target(float(np.abs(difference)))
    assert equal
    assert equal == True  # noqa: E712
    assert bool(equal) is True
    assert_array_equal(array, array, tol)


@example(
    arrays=(np.array([1, 2, 3]), np.array([1, 2, 3])),
    unit_tol=1e-4,
    with_sign=False,
)
@given(
    arrays=array_shapes().flatmap(
        lambda shape: st.tuples(
            arrays(
                dtype=st.one_of(floating_dtypes(), integer_dtypes()),
                shape=shape,
            ),
            arrays(
                dtype=st.one_of(floating_dtypes(), integer_dtypes()),
                shape=shape,
            ),
        )
    ),
    unit_tol=st.floats(1e-4, 1.0),
    with_sign=st.booleans(),
)
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract")
def test_array_equal_two_arrays(arrays, unit_tol, with_sign):
    array_a, array_b = arrays
    equal = array_equal(array_a, array_b, unit_tol, with_sign=with_sign)
    equal_flipped = array_equal(
        array_b, array_a, unit_tol, with_sign=with_sign
    )
    note(equal)
    difference = equal.compute_difference()
    a, b = (
        (array_a, array_b) if with_sign else (np.abs(array_a), np.abs(array_b))
    )
    expect_equal = np.sum(np.abs(a - b) > unit_tol) / array_a.size < 1e-4
    if expect_equal:
        assert_array_equal(array_a, array_b, unit_tol, with_sign=with_sign)
        assert equal
        assert bool(equal) is True
        assert equal == True  # noqa: E712
        assert True == equal  # noqa: E712
        assert equal != False  # noqa: E712
        assert False != equal  # noqa: E712
        assert equal_flipped
        assert bool(equal_flipped) is True
        assert equal_flipped == True  # noqa: E712
        assert True == equal_flipped  # noqa: E712
        assert equal_flipped != False  # noqa: E712
        assert False != equal_flipped  # noqa: E712
    else:
        with pytest.raises(AssertionError):
            assert_array_equal(array_a, array_b, unit_tol, with_sign=with_sign)
        assert not equal
        assert bool(equal) is not True
        assert equal != True  # noqa: E712
        assert True != equal  # noqa: E712
        assert equal == False  # noqa: E712
        assert False == equal  # noqa: E712
        assert difference != 0
