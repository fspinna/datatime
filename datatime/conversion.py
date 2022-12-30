import numpy as np
import awkward as ak
from numpy.typing import NDArray
from typing import Any, List
import pandas as pd


def has_same_number_of_signals(X: ak.Array) -> bool:
    return len(np.unique(ak.num(X, axis=1))) == 1


def is_univariate(X: ak.Array) -> bool:
    return has_same_number_of_signals(X) and np.unique(ak.num(X, axis=1))[0] == 1


def is_multivariate(X: ak.Array) -> bool:
    return has_same_number_of_signals(X) and np.unique(ak.num(X, axis=1))[0] > 1


def has_equal_length_signals(X: ak.Array) -> bool:
    return has_same_number_of_signals(X) and len(np.unique(ak.num(X, axis=2))) == 1


def _awkward_to_pyts(X: ak.Array) -> NDArray[Any]:
    assert is_univariate(X), "Pyts only supports univariate time series."
    assert has_equal_length_signals(
        X
    ), "Pyts only supports time series having equal length."
    return np.squeeze(X.to_numpy(), axis=1)


def _awkward_to_tslearn(X: ak.Array) -> NDArray[Any]:
    assert has_equal_length_signals(
        X
    ), "Tslearn only supports time series having equal length."
    return np.swapaxes(X.to_numpy(), 1, 2)


def _awkward_to_sktime(X: ak.Array) -> pd.DataFrame:
    assert has_same_number_of_signals(
        X
    ), "Sktime only supports multivariate time series having an equal number of signals."
    df = pd.DataFrame()
    n, k = len(X), len(X[0])
    for j in range(k):
        df[str(j)] = [pd.Series(X[i, j, :].to_numpy()) for i in range(n)]
    return df


def awkward_to_pyts(*args: ak.Array) -> List[NDArray[Any]]:
    return [_awkward_to_pyts(X) for X in args]


def awkward_to_tslearn(*args: ak.Array) -> List[NDArray[Any]]:
    return [_awkward_to_tslearn(X) for X in args]


def awkward_to_sktime(*args: ak.Array) -> List[pd.DataFrame]:
    return [_awkward_to_sktime(X) for X in args]


if __name__ == "__main__":
    pass
