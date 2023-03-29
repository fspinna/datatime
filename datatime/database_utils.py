import json
from typing import Union, Any, Optional, List, Tuple, Dict
from numpy.typing import NDArray
import awkward as ak
import numpy as np
import pandas as pd
import pathlib
from datatime.download_utils import download_dataset
from datatime.classes import (
    TimeSeriesClassificationDataset,
    TimeSeriesRegressionDataset,
    TimeSeriesForecastingDataset,
    TimeSeriesMultioutputDataset,
)
from datatime.utils import get_project_root, get_default_dataset_path, fill_none


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


def datasets_table(tasks: Optional[List[str]] = None) -> pd.DataFrame:
    df = (
        pd.read_csv(get_project_root() / "database.csv")
        .drop(["file"], axis=1)
        .drop_duplicates()
    )
    if tasks is None:
        return df.sort_values(["dataset"])
    else:
        return df[df["task"].isin(tasks)].sort_values(["dataset"])


def datasets_list(tasks: Optional[List[str]] = None):
    return datasets_table(tasks=tasks)["dataset"].to_list()


def cached_datasets_dict(root: Optional[pathlib.Path] = None) -> Dict[str, List[str]]:
    if root is None:
        root = get_project_root() / "cached_datasets"
    root.mkdir(parents=True, exist_ok=True)
    tasks = sorted([d.name for d in root.iterdir() if d.is_dir()])
    d = dict()
    for task in tasks:
        datasets = sorted([d.name for d in (root / task).iterdir() if d.is_dir()])
        d[task] = datasets
    return d


def is_cached(dataset_name: str, task: str) -> bool:
    d = cached_datasets_dict()
    if task not in d:
        return False
    if dataset_name in d[task]:
        return True
    else:
        return False


def X_info(X: ak.Array) -> Tuple[int, int, int, int, bool, bool]:
    m_max = ak.max(ak.ravel(ak.count(X, axis=2)))
    m_min = ak.min(ak.ravel(ak.count(X, axis=2)))
    k = len(X[0])
    n = len(X)
    m_constant = m_max == m_min
    missing_values = np.any(np.isnan(ak.ravel(X)))
    return n, k, m_max, m_min, m_constant, missing_values


def load_dataset(
    name: str, nan_value: float = np.nan
) -> Union[
    TimeSeriesClassificationDataset,
    TimeSeriesRegressionDataset,
    TimeSeriesForecastingDataset,
    TimeSeriesMultioutputDataset,
]:
    """
    Load a time series dataset.

    :param name: The name of the dataset to load.
    :param nan_value: The value that represents a missing value.
    :return: A TimeSeriesDataset object.
    """
    d = datasets_table()
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
    elif dataset_type == "multioutput":
        X_train, Y_train, X_test, Y_test = load_multioutput_dataset(
            name, nan_value=nan_value
        )
        return TimeSeriesMultioutputDataset(
            X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, name=name
        )
    else:
        raise ValueError("Dataset type not implemented.")


def load_classification_dataset(
    name: str,
    nan_value: float = np.nan,
    origin: str = "gdrive",
    path: Optional[str] = None,
) -> Tuple[ak.Array, NDArray[Any], ak.Array, NDArray[Any], Dict[int, str]]:
    if path is None:
        path = get_default_dataset_path(dataset_name=name, task="classification")
        if not is_cached(dataset_name=name, task="classification"):
            download_dataset(name=name, origin=origin)
    else:
        path = pathlib.Path(path)

    X_train = ak.from_json(path / (name + "__X_train.json"))
    X_test = ak.from_json(path / (name + "__X_test.json"))
    X_train, X_test = fill_none(X_train, X_test, replace_with=nan_value)
    y_train = np.array(ak.from_json(path / (name + "__y_train.json")))
    y_test = np.array(ak.from_json(path / (name + "__y_test.json")))
    with open(path / (name + "__labels.json")) as out:
        labels = {int(key): value for (key, value) in json.load(out).items()}
    return X_train, y_train, X_test, y_test, labels


def load_regression_dataset(
    name: str, nan_value: float = np.nan, origin="gdrive"
) -> Tuple[ak.Array, NDArray[Any], ak.Array, NDArray[Any]]:
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
) -> Tuple[ak.Array, ak.Array]:
    path = get_default_dataset_path(dataset_name=name, task="forecasting")

    if not is_cached(dataset_name=name, task="forecasting"):
        download_dataset(name=name, origin=origin)

    X = ak.from_json(path / (name + "__X.json"))
    Y = ak.from_json(path / (name + "__Y.json"))
    X, Y = fill_none(X, Y, replace_with=nan_value)
    return X, Y


def load_multioutput_dataset(
    name: str,
    nan_value: float = np.nan,
    origin="gdrive",
    path: Optional[str] = None,
) -> Tuple[ak.Array, pd.DataFrame, ak.Array, pd.DataFrame]:
    if path is None:
        path = get_default_dataset_path(dataset_name=name, task="multioutput")
        if not is_cached(dataset_name=name, task="multioutput"):
            download_dataset(name=name, origin=origin)
    else:
        path = pathlib.Path(path)

    X_train = ak.from_json(path / (name + "__X_train.json"))
    X_test = ak.from_json(path / (name + "__X_test.json"))
    X_train, X_test = fill_none(X_train, X_test, replace_with=nan_value)
    Y_train = pd.read_csv(path / (name + "__Y_train.csv"))
    Y_test = pd.read_csv(path / (name + "__Y_test.csv"))
    return X_train, Y_train, X_test, Y_test
