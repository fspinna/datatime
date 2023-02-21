import pandas as pd
import pytest
import awkward as ak
import numpy as np
from datatime.conversion import (
    has_same_number_of_signals,
    is_univariate,
    is_multivariate,
    has_equal_length_signals,
    awkward_to_pyts,
    awkward_to_sktime,
    awkward_to_tslearn,
    sktime_to_awkward,
    awkward_to_flat,
    pyts_to_awkward,
)

SIGNAL_LENGTH_1 = 10
SIGNAL_LENGTH_2 = 20
SIGNAL_LENGTH_3 = 11

# time series dataset with one time series containing one element
UNIVARIATE_SINGLETON = ak.Array([[[1]]])
UNIVARIATE_SINGLETON_EXPECTED_PYTS = np.array([[1]])

# univariate time series dataset with equal length time series
UNIVARIATE_SAME_NUMBER_OF_SIGNALS = ak.Array(
    [[np.arange(SIGNAL_LENGTH_1)], [np.arange(SIGNAL_LENGTH_1)]]
)
UNIVARIATE_EXPECTED_PYTS = np.array(
    [np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_1)]
)

# time series dataset containing missing some time series (different from having empty time series)
UNIVARIATE_DIFFERENT_NUMBER_OF_SIGNALS = ak.Array([[np.arange(SIGNAL_LENGTH_1)], []])

# univariate time series dataset with different length time series
UNIVARIATE_DIFFERENT_SIGNAL_LENGTH = ak.Array(
    [
        [np.arange(SIGNAL_LENGTH_1)],
        [np.arange(SIGNAL_LENGTH_2)],
    ]
)

# multivariate time series dataset without missing signals
MULTIVARIATE_SAME_NUMBER_OF_SIGNALS = ak.Array(
    [
        [np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_1)],
        [np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_1)],
    ]
)
MULTIVARIATE_EXPECTED_TSLEARN = np.swapaxes(
    [
        [np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_1)],
        [np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_1)],
    ],
    1,
    2,
)


# multivariate time series dataset missing some signals
MULTIVARIATE_DIFFERENT_NUMBER_OF_SIGNALS = ak.Array(
    [
        [np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_1)],
        [np.arange(SIGNAL_LENGTH_1)],
    ]
)

# multivariate time series dataset having signals with different lengths
MULTIVARIATE_DIFFERENT_SIGNAL_LENGTH = ak.Array(
    [
        [np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_2)],
        [np.arange(SIGNAL_LENGTH_3), np.arange(SIGNAL_LENGTH_1)],
    ]
)
MULTIVARIATE_EXPECTED_SKTIME = pd.DataFrame(
    {
        "0": [
            pd.Series(np.arange(SIGNAL_LENGTH_1)),
            pd.Series(np.arange(SIGNAL_LENGTH_3)),
        ],
        "1": [
            pd.Series(np.arange(SIGNAL_LENGTH_2)),
            pd.Series(np.arange(SIGNAL_LENGTH_1)),
        ],
    }
)
MULTIVARIATE_EXPECTED_FLAT = ak.Array(
    [
        np.concatenate([np.arange(SIGNAL_LENGTH_1), np.arange(SIGNAL_LENGTH_2)]),
        np.concatenate([np.arange(SIGNAL_LENGTH_3), np.arange(SIGNAL_LENGTH_1)]),
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


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (UNIVARIATE_EXPECTED_PYTS, UNIVARIATE_SAME_NUMBER_OF_SIGNALS),
        (UNIVARIATE_SINGLETON_EXPECTED_PYTS, UNIVARIATE_SINGLETON),
    ],
)
def test__pyts_to_awkward_conversion(test_input, expected):
    X = pyts_to_awkward(test_input)
    assert np.array_equal(X, expected)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (UNIVARIATE_SAME_NUMBER_OF_SIGNALS, UNIVARIATE_EXPECTED_PYTS),
        (UNIVARIATE_SINGLETON, UNIVARIATE_SINGLETON_EXPECTED_PYTS),
    ],
)
def test__awkward_to_pyts_conversion(test_input, expected):
    X_pyts = awkward_to_pyts(test_input)
    assert np.all(X_pyts == expected)


@pytest.mark.parametrize(
    "test_input",
    [UNIVARIATE_DIFFERENT_SIGNAL_LENGTH, MULTIVARIATE_SAME_NUMBER_OF_SIGNALS],
)
def test__awkward_to_pyts_exception(test_input):
    with pytest.raises(Exception):
        awkward_to_pyts(test_input)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (MULTIVARIATE_DIFFERENT_SIGNAL_LENGTH, MULTIVARIATE_EXPECTED_SKTIME),
    ],
)
def test__awkward_to_sktime_conversion(test_input, expected):
    X_sktime = awkward_to_sktime(test_input)
    for i in range(len(expected)):
        for j in range(len(expected.columns)):
            assert np.all(X_sktime.iloc[i, j].values == expected.iloc[i, j].values)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (MULTIVARIATE_EXPECTED_SKTIME, MULTIVARIATE_DIFFERENT_SIGNAL_LENGTH),
    ],
)
def test__sktime_to_awkward_conversion(test_input, expected):
    X = sktime_to_awkward(test_input)
    assert ak.all(X == expected)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (MULTIVARIATE_SAME_NUMBER_OF_SIGNALS, MULTIVARIATE_EXPECTED_TSLEARN),
    ],
)
def test__awkward_to_tslearn_conversion(test_input, expected):
    X_tslearn = awkward_to_tslearn(test_input)
    assert np.all(X_tslearn == expected)


@pytest.mark.parametrize(
    "test_input",
    [UNIVARIATE_DIFFERENT_SIGNAL_LENGTH, MULTIVARIATE_DIFFERENT_SIGNAL_LENGTH],
)
def test__awkward_to_tslearn_exception(test_input):
    with pytest.raises(Exception):
        awkward_to_tslearn(test_input)


@pytest.mark.parametrize(
    "test_input,expected",
    [(MULTIVARIATE_DIFFERENT_SIGNAL_LENGTH, MULTIVARIATE_EXPECTED_FLAT)],
)
def test__awkward_to_flat_conversion(test_input, expected):
    X_flat = awkward_to_flat(test_input)
    assert np.all(X_flat == expected)


if __name__ == "__main__":
    pytest.main()
