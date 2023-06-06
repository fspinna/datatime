from typing import List, Union, Dict, Any, Tuple

import awkward as ak
import numpy as np
import pandas as pd

from downtime import load_dataset
from downtime.classes import (
    TimeSeriesClassificationDataset,
    TimeSeriesRegressionDataset,
    TimeSeriesForecastingDataset,
)


def datasets_info(names: List[str]) -> pd.DataFrame:
    info_df = pd.DataFrame()
    for name in names:
        d = load_dataset(name)
        infos = pd.DataFrame(dataset_info(d), index=[name])
        info_df = pd.concat([info_df, infos], axis=0)
    return info_df


def dataset_info(
    dataset: Union[
        TimeSeriesClassificationDataset,
        TimeSeriesRegressionDataset,
        TimeSeriesForecastingDataset,
    ]
) -> Dict[str, Any]:
    if isinstance(dataset, TimeSeriesClassificationDataset):
        X_train, y_train, X_test, y_test = dataset()
        (
            n_train,
            k_train,
            m_max_train,
            m_min_train,
            m_constant_train,
            missing_values_train,
        ) = X_info(X_train)
        (
            n_test,
            k_test,
            m_max_test,
            m_min_test,
            m_constant_test,
            missing_values_test,
        ) = X_info(X_test)
        n_labels_train = len(np.unique(y_train))
        n_labels_test = len(np.unique(y_test))
        return {
            "n_train": n_train,
            "k_train": k_train,
            "m_min_train": m_min_train,
            "m_max_train": m_max_train,
            "m_constant_train": m_constant_train,
            "n_labels_train": n_labels_train,
            "n_test": n_test,
            "k_test": k_test,
            "m_min_test": m_min_test,
            "m_max_test": m_max_test,
            "m_constant_test": m_constant_test,
            # FIXME: m_constant does not consider constant train and test but with different lengths w.r.t each other
            "m_constant": m_constant_train and m_constant_test,
            "n_labels_test": n_labels_test,
            "missing_values_train": missing_values_train,
            "missing_values_test": missing_values_test,
            "missing_values": missing_values_train or missing_values_test,
            "m_constant_no_missing": (m_constant_train and m_constant_test)
            and not (missing_values_train or missing_values_test),
        }
    elif isinstance(dataset, TimeSeriesRegressionDataset):
        X_train, y_train, X_test, y_test = dataset()
        (
            n_train,
            k_train,
            m_max_train,
            m_min_train,
            m_constant_train,
            missing_values_train,
        ) = X_info(X_train)
        (
            n_test,
            k_test,
            m_max_test,
            m_min_test,
            m_constant_test,
            missing_values_test,
        ) = X_info(X_test)
        return {
            "n_train": n_train,
            "k_train": k_train,
            "m_min_train": m_min_train,
            "m_max_train": m_max_train,
            "m_constant_train": m_constant_train,
            "n_test": n_test,
            "k_test": k_test,
            "m_min_test": m_min_test,
            "m_max_test": m_max_test,
            "m_constant_test": m_constant_test,
            "m_constant": m_constant_train and m_constant_test,
            "missing_values_train": missing_values_train,
            "missing_values_test": missing_values_test,
            "missing_values": missing_values_train or missing_values_test,
            "m_constant_no_missing": (m_constant_train and m_constant_test)
            and not (missing_values_train or missing_values_test),
        }
    elif isinstance(dataset, TimeSeriesForecastingDataset):
        X, Y = dataset()
        (
            n_X,
            k_X,
            m_max_X,
            m_min_X,
            m_constant_X,
            missing_values_X,
        ) = X_info(X)
        (
            n_Y,
            k_Y,
            m_max_Y,
            m_min_Y,
            m_constant_Y,
            missing_values_Y,
        ) = X_info(Y)
        return {
            "n_X": n_X,
            "k_X": k_X,
            "m_min_X": m_min_X,
            "m_max_X": m_max_X,
            "m_constant_X": m_constant_X,
            "n_Y": n_Y,
            "k_Y": k_Y,
            "m_min_Y": m_min_Y,
            "m_max_Y": m_max_Y,
            "m_constant_Y": m_constant_Y,
            "m_constant": m_constant_X and m_constant_Y,
            "missing_values_X": missing_values_X,
            "missing_values_Y": missing_values_Y,
            "missing_values": missing_values_X or missing_values_Y,
            "m_constant_no_missing": (m_constant_X and m_constant_Y)
            and not (missing_values_X or missing_values_Y),
        }
    else:
        raise Exception(ValueError)


def X_info(X: ak.Array) -> Tuple[int, int, int, int, bool, bool]:
    m_max = ak.max(ak.ravel(ak.count(X, axis=2)))
    m_min = ak.min(ak.ravel(ak.count(X, axis=2)))
    k = len(X[0])
    n = len(X)
    m_constant = m_max == m_min
    missing_values = np.any(np.isnan(ak.ravel(X)))
    return n, k, m_max, m_min, m_constant, missing_values
