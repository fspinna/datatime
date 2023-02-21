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


def awkward_to_pyts(X: ak.Array) -> NDArray[Any]:
    assert is_univariate(X), "Pyts only supports univariate time series."
    assert has_equal_length_signals(
        X
    ), "Pyts only supports time series having equal length."
    return np.asarray(np.squeeze(X.to_numpy(), axis=1))


def pyts_to_awkward(X: NDArray) -> ak.Array:
    return ak.Array(X[:, np.newaxis, :])


def awkward_to_tslearn(X: ak.Array) -> NDArray[Any]:
    assert has_equal_length_signals(
        X
    ), "Tslearn only supports time series having equal length."
    return np.swapaxes(X.to_numpy(), 1, 2)


def awkward_to_sktime(X: ak.Array) -> pd.DataFrame:
    assert has_same_number_of_signals(
        X
    ), "Sktime only supports multivariate time series having an equal number of signals."
    df = dict()  # pd.DataFrame()
    n, k = len(X), len(X[0])
    for j in range(k):
        df[str(j)] = [pd.Series(X[i, j, :].to_numpy()) for i in range(n)]
    return pd.DataFrame(df)


def sktime_to_awkward(X: pd.DataFrame) -> ak.Array:
    builder = ak.ArrayBuilder()
    for row in range(X.shape[0]):
        builder.begin_list()
        for column in range(X.shape[1]):
            builder.begin_list()
            for value in X.iloc[row, column]:
                builder.real(value)
            builder.end_list()
        builder.end_list()
    return builder.snapshot()


def awkward_to_flat(X: ak.Array) -> ak.Array:
    return ak.flatten(X, axis=2)


if __name__ == "__main__":
    pass
