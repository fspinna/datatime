import json
from typing import Union

import awkward as ak
import numpy as np
import pandas as pd

from datatime.download_utils import download_dataset
from datatime.main import TimeSeriesClassificationDataset, TimeSeriesRegressionDataset, \
    TimeSeriesForecastingDataset
from datatime.utils import get_project_root, get_default_dataset_path, fill_none


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
    root.mkdir(parents=True, exist_ok=True)
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


def load_classification_dataset(
    name: str, nan_value: float = np.nan, origin="gdrive"
) -> tuple[ak.Array, np.array, ak.Array, np.array, dict]:
    path = get_default_dataset_path(dataset_name=name, task="classification")

    if not is_cached(dataset_name=name, task="classification"):
        download_dataset(name=name, origin=origin)

    X_train = ak.from_json(path / (name + "__X_train.json"))
    X_test = ak.from_json(path / (name + "__X_test.json"))
    X_train, X_test = fill_none(X_train, X_test, replace_with=nan_value)
    y_train = np.array(ak.from_json(path / (name + "__y_train.json")))
    y_test = np.array(ak.from_json(path / (name + "__y_test.json")))
    with open(path / (name + "__labels.json")) as l:
        labels = {int(key): value for (key, value) in json.load(l).items()}
    return X_train, y_train, X_test, y_test, labels


def load_regression_dataset(
    name: str, nan_value: float = np.nan, origin="gdrive"
) -> tuple[ak.Array, np.array, ak.Array, np.array]:
    path = get_default_dataset_path(dataset_name=name, task="regression")

    if not is_cached(dataset_name=name, task="regression"):
        download_dataset(name=name, origin=origin)

    X_train = ak.from_json(path / (name + "__X_train.json"))
    X_test = ak.from_json(path / (name + "__X_test.json"))
    X_train, X_test = fill_none(X_train, X_test, replace_with=nan_value)
    y_train = np.array(ak.from_json(path / (name + "__y_train.json")))
    y_test = np.array(ak.from_json(path / (name + "__y_test.json")))
    return X_train, y_train, X_test, y_test


def load_forecasting_dataset(
    name: str, nan_value: float = np.nan, origin="gdrive"
) -> tuple[ak.Array, ak.Array]:
    path = get_default_dataset_path(dataset_name=name, task="forecasting")

    if not is_cached(dataset_name=name, task="forecasting"):
        download_dataset(name=name, origin=origin)

    X = ak.from_json(path / (name + "__X.json"))
    Y = ak.from_json(path / (name + "__Y.json"))
    X, Y = fill_none(X, Y, replace_with=nan_value)
    return X, Y
