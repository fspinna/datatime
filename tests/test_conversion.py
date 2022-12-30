import pandas as pd
import pytest
import awkward as ak
import numpy as np
from datatime.conversion import (
    has_same_number_of_signals,
    is_univariate,
    is_multivariate,
    has_equal_length_signals,
    _awkward_to_pyts,
    _awkward_to_sktime,
)

SIGNAL_LENGTH_1 = 10
SIGNAL_LENGTH_2 = 20

UNIVARIATE_SINGLETON = ak.Array([[[1]]])
UNIVARIATE_SINGLETON_EXPECTED_PYTS = np.array([[1]])
UNIVARIATE_SAME_NUMBER_OF_SIGNALS = ak.Array(
    [[np.arange(SIGNAL_LENGTH_1)], [np.arange(SIGNAL_LENGTH_1)]]
)
UNIVARIATE_EXPECTED_PYTS = np.array(
    [np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_1)]
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
MULTIVARIATE_EXPECTED_SKTIME = pd.DataFrame(
    {
        "0": [
            pd.Series(np.arange(SIGNAL_LENGTH_1)),
            pd.Series(np.arange(SIGNAL_LENGTH_2)),
        ],
        "1": [
            pd.Series(np.arange(SIGNAL_LENGTH_2)),
            pd.Series(np.arange(SIGNAL_LENGTH_1)),
        ],
    }
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


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (UNIVARIATE_SAME_NUMBER_OF_SIGNALS, UNIVARIATE_EXPECTED_PYTS),
        (UNIVARIATE_SINGLETON, UNIVARIATE_SINGLETON_EXPECTED_PYTS),
    ],
)
def test__awkward_to_pyts_conversion(test_input, expected):
    X_pyts = _awkward_to_pyts(test_input)
    assert np.all(X_pyts == expected)


@pytest.mark.parametrize(
    "test_input",
    [UNIVARIATE_DIFFERENT_SIGNAL_LENGTH, MULTIVARIATE_SAME_NUMBER_OF_SIGNALS],
)
def test__awkward_to_pyts_exception(test_input):
    with pytest.raises(Exception):
        _awkward_to_pyts(test_input)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (MULTIVARIATE_DIFFERENT_SIGNAL_LENGTH, MULTIVARIATE_EXPECTED_SKTIME),
    ],
)
def test__awkward_to_sktime_conversion(test_input, expected):
    X_sktime = _awkward_to_sktime(test_input)
    for i in range(len(expected)):
        for j in range(len(expected.columns)):
            assert np.all(X_sktime.iloc[i, j].values == expected.iloc[i, j].values)


if __name__ == "__main__":
    pytest.main()
