import numpy as np
import awkward as ak
from numpy.typing import NDArray
from typing import Any
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
    assert is_univariate(X)
    assert has_equal_length_signals(X)
    return np.squeeze(X.to_numpy(), axis=1)


def _awkward_to_tslearn(X: ak.Array) -> NDArray[Any]:
    assert has_equal_length_signals(X)
    return np.swapaxes(X.to_numpy(), 1, 2)


def _awkward_to_sktime(X: ak.Array) -> pd.DataFrame:
    assert has_same_number_of_signals(X)
    X_numpy = X.to_numpy()
    df = pd.DataFrame()
    n, k, m = X.shape
    for (j,) in range(k):
        df[j] = [pd.Series(X_numpy[instance, j, :]) for instance in range(n)]
    return df


def awkward_to_pyts(*args: ak.Array) -> list[NDArray[Any]]:
    return [_awkward_to_pyts(X) for X in args]


def awkward_to_tslearn(*args: ak.Array) -> list[NDArray[Any]]:
    return [_awkward_to_tslearn(X) for X in args]


def awkward_to_sktime(*args: ak.Array) -> list[pd.DataFrame]:
    return [_awkward_to_sktime(X) for X in args]
