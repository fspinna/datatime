import pytest
import awkward as ak
import numpy as np
from datatime.conversion import (
    has_same_number_of_signals,
    is_univariate,
    is_multivariate,
    has_equal_length_signals,
)

SIGNAL_LENGTH_1 = 10
SIGNAL_LENGTH_2 = 20
UNIVARIATE_SAME_NUMBER_OF_SIGNALS = ak.Array(
    [[np.arange(SIGNAL_LENGTH_1)], [np.arange(SIGNAL_LENGTH_1)]]
)
UNIVARIATE_DIFFERENT_NUMBER_OF_SIGNALS = ak.Array([[np.arange(SIGNAL_LENGTH_1)], []])
UNIVARIATE_DIFFERENT_SIGNAL_LENGTH = ak.Array(
    [
        [np.arange(SIGNAL_LENGTH_1)],
        [np.arange(SIGNAL_LENGTH_2)],
    ]
)
MULTIVARIATE_SAME_NUMBER_OF_SIGNALS = ak.Array(
    [
        [np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_1)],
        [np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_1)],
    ]
)
MULTIVARIATE_DIFFERENT_NUMBER_OF_SIGNALS = ak.Array(
    [
        [np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_1)],
        [np.arange(SIGNAL_LENGTH_1)],
    ]
)
MULTIVARIATE_DIFFERENT_SIGNAL_LENGTH = ak.Array(
    [
        [np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_2)],
        [np.arange(SIGNAL_LENGTH_2), np.arange(SIGNAL_LENGTH_1)],
    ]
)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (MULTIVARIATE_SAME_NUMBER_OF_SIGNALS, True),
        (MULTIVARIATE_DIFFERENT_NUMBER_OF_SIGNALS, False),
        (UNIVARIATE_SAME_NUMBER_OF_SIGNALS, True),
        (UNIVARIATE_DIFFERENT_NUMBER_OF_SIGNALS, False),
    ],
)
def test_has_same_number_of_signals(test_input, expected):
    assert has_same_number_of_signals(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (MULTIVARIATE_SAME_NUMBER_OF_SIGNALS, False),
        (UNIVARIATE_SAME_NUMBER_OF_SIGNALS, True),
        (UNIVARIATE_DIFFERENT_SIGNAL_LENGTH, True),
    ],
)
def test_is_univariate(test_input, expected):
    assert is_univariate(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (MULTIVARIATE_SAME_NUMBER_OF_SIGNALS, True),
        (UNIVARIATE_SAME_NUMBER_OF_SIGNALS, False),
        (MULTIVARIATE_DIFFERENT_SIGNAL_LENGTH, True),
    ],
)
def test_is_multivariate(test_input, expected):
    assert is_multivariate(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (MULTIVARIATE_SAME_NUMBER_OF_SIGNALS, True),
        (UNIVARIATE_SAME_NUMBER_OF_SIGNALS, True),
        (MULTIVARIATE_DIFFERENT_SIGNAL_LENGTH, False),
        (UNIVARIATE_DIFFERENT_SIGNAL_LENGTH, False),
    ],
)
def test_has_equal_length_signals(test_input, expected):
    assert has_equal_length_signals(test_input) == expected


if __name__ == "__main__":
    pytest.main()
