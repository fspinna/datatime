from typing import Union

import awkward as ak
import numpy as np
import pandas as pd

from datatime.loader import load_classification_dataset, load_regression_dataset, load_forecasting_dataset
from datatime.main import TimeSeriesClassificationDataset, TimeSeriesRegressionDataset, \
    TimeSeriesForecastingDataset
from datatime.utils import get_project_root


def datasets_info(names):
    info_df = pd.DataFrame()
    for name in names:
        d = load_dataset(name)
        infos = pd.DataFrame(dataset_info(d), index=[name])
        info_df = pd.concat([info_df, infos], axis=0)
    return info_df


def dataset_info(d: Union[
    TimeSeriesClassificationDataset,
    TimeSeriesRegressionDataset,
    TimeSeriesForecastingDataset,
]):
    if isinstance(d, TimeSeriesClassificationDataset):
        X_train, y_train, X_test, y_test = d()
        n_train, k_train, m_max_train, m_min_train, m_constant_train = X_info(X_train)
        n_test, k_test, m_max_test, m_min_test, m_constant_test = X_info(X_test)
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
            "n_labels_test": n_labels_test,
        }
    elif isinstance(d, TimeSeriesRegressionDataset):
        X_train, y_train, X_test, y_test = d()
        n_train, k_train, m_max_train, m_min_train, m_constant_train = X_info(X_train)
        n_test, k_test, m_max_test, m_min_test, m_constant_test = X_info(X_test)
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
        }


def datasets_df():
    df = pd.read_csv(get_project_root() / "database.csv").drop(["file"], axis=1).drop_duplicates()
    return df.sort_values(["dataset"])


def cached_datasets_dict(root=None):
    if root is None:
        root = get_project_root() / "cached_datasets"
    tasks = sorted([d.name for d in root.iterdir() if d.is_dir()])
    d = dict()
    for task in tasks:
        datasets = sorted([d.name for d in (root / task).iterdir() if d.is_dir()])
        d[task] = datasets
    return d


def is_cached(dataset_name, task):
    d = cached_datasets_dict()
    if task not in d:
        return False
    if dataset_name in d[task]:
        return True
    else:
        return False


def X_info(X: ak.Array):
    m_max = ak.max(ak.ravel(ak.count(X, axis=2)))
    m_min = ak.min(ak.ravel(ak.count(X, axis=2)))
    k = len(X[0])
    n = len(X)
    m_constant = m_max == m_min
    return n, k, m_max, m_min, m_constant


def load_dataset(
    name: str, nan_value: float = np.nan
) -> Union[
    TimeSeriesClassificationDataset,
    TimeSeriesRegressionDataset,
    TimeSeriesForecastingDataset,
]:
    d = datasets_df()
    dataset_type = d[d["dataset"] == name]["task"].values[0]
    if dataset_type == "classification":
        X_train, y_train, X_test, y_test, labels = load_classification_dataset(
            name, nan_value=nan_value
        )
        return TimeSeriesClassificationDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            labels=labels,
            name=name,
        )
    elif dataset_type == "regression":
        X_train, y_train, X_test, y_test = load_regression_dataset(
            name, nan_value=nan_value
        )
        return TimeSeriesRegressionDataset(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, name=name
        )
    elif dataset_type == "forecasting":
        X, Y = load_forecasting_dataset(name, nan_value=nan_value)
        return TimeSeriesForecastingDataset(X=X, Y=Y, name=name)
    else:
        raise ValueError("Dataset type not implemented.")
