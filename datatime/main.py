import json
import pathlib
import awkward as ak
import numpy as np
from typing import Optional
import pandas as pd
from datatime.utils import get_project_root, fill_none


class TimeSeriesDataset(object):
    pass


class TimeSeriesClassificationDataset(TimeSeriesDataset):
    def __init__(
            self,
            X_train: ak.Array,
            y_train: np.array,
            X_test: ak.Array,
            y_test: np.array,
            labels: Optional[dict] = None,
            name: str = "",
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.labels = labels
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def __str__(self):
        try:
            return (
                    "Dataset Name: %s\nTask: classification\nX_train: %s\nX_test: %s\ny_train: %s\ny_test: %s\nLabel "
                    "Encoding: %s"
                    % (
                        self.name,
                        np.array(self.X_train).shape,
                        np.array(self.X_test).shape,
                        self.y_train.shape,
                        self.y_test.shape,
                        self.labels,
                    )
            )
        except ValueError:
            return (
                    "Dataset Name: %s\nTask: classification\nX_train: %s\nX_test: %s\ny_train: %s\ny_test: %s\nLabel "
                    "Encoding: %s"
                    % (
                        self.name,
                        self.X_train.type,
                        self.X_test.type,
                        self.y_train.shape,
                        self.y_test.shape,
                        self.labels,
                    )
            )

    def map_labels(self, y):
        return np.vectorize(self.labels.get)(y)


class TimeSeriesRegressionDataset(TimeSeriesDataset):
    def __init__(
            self,
            X_train: ak.Array,
            y_train: np.array,
            X_test: ak.Array,
            y_test: np.array,
            name: str = "",
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def __str__(self):
        try:
            return (
                    "Dataset Name: %s\nTask: regression\nX_train: %s\nX_test: %s\ny_train: %s\ny_test: %s"
                    % (
                        self.name,
                        np.array(self.X_train).shape,
                        np.array(self.X_test).shape,
                        self.y_train.shape,
                        self.y_test.shape,
                    )
            )
        except ValueError:
            return (
                    "Dataset Name: %s\nTask: regression\nX_train: %s\nX_test: %s\ny_train: %s\ny_test: %s"
                    % (
                        self.name,
                        self.X_train.type,
                        self.X_test.type,
                        self.y_train.shape,
                        self.y_test.shape,
                    )
            )


class TimeSeriesForecastingDataset(TimeSeriesDataset):
    def __init__(self, X: ak.Array, Y: ak.Array, name: str = ""):
        self.X = X
        self.Y = Y
        self.XY = self._concatenate_X_Y()
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.X, self.Y

    def __str__(self):
        try:
            return "Dataset Name: %s\nTask: forecasting\nX: %s\nY: %s" % (
                self.name,
                np.array(self.X).shape,
                np.array(self.Y).shape,
            )
        except ValueError:
            return "Dataset Name: %s\nTask: forecasting\nX: %s\nY: %s" % (
                self.name,
                self.X.type,
                self.Y.type,
            )

    def _concatenate_X_Y(self):
        return ak.concatenate([self.X, self.Y], axis=2)


def dataset_info(d: TimeSeriesDataset):
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
            "n_labels_test": n_labels_test
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


def dataset_dict():
    root = get_project_root()
    classification = (root / "datasets" / "classification").glob("**/*")
    regression = (root / "datasets" / "regression").glob("**/*")
    forecasting = (root / "datasets" / "forecasting").glob("**/*")
    d = {
        "classification": sorted([x.name for x in classification if x.is_dir()]),
        "regression": sorted([x.name for x in regression if x.is_dir()]),
        "forecasting": sorted([x.name for x in forecasting if x.is_dir()]),
    }
    return d


if __name__ == "__main__":
    # d = load_dataset("CBF")
    # print(d)
    print(pathlib.Path(__file__).parent)
    print(dataset_info(dataset_dict()["classification"]))


def X_info(X: ak.Array):
    m_max = ak.max(ak.ravel(ak.count(X, axis=2)))
    m_min = ak.min(ak.ravel(ak.count(X, axis=2)))
    k = len(X[0])
    n = len(X)
    m_constant = m_max == m_min
    return n, k , m_max, m_min, m_constant


def datasets_info(names):
    info_df = pd.DataFrame()
    for name in names:
        d = load_dataset(name)
        infos = pd.DataFrame(dataset_info(d), index=[name])
        info_df = pd.concat([info_df, infos], axis=0)
    return info_df


def load_classification_dataset(
        name: str, nan_value: float = np.nan
) -> tuple[ak.Array, np.array, ak.Array, np.array, np.array]:
    root = get_project_root()
    path = root / "datasets" / "classification" / name
    X_train = ak.from_json(path / (name + "__X_train.json"))
    X_test = ak.from_json(path / (name + "__X_test.json"))
    X_train, X_test = fill_none(X_train, X_test, replace_with=nan_value)
    y_train = np.array(ak.from_json(path / (name + "__y_train.json")))
    y_test = np.array(ak.from_json(path / (name + "__y_test.json")))
    with open(path / (name + "__labels.json")) as l:
        labels = {int(key): value for (key, value) in json.load(l).items()}
    return X_train, y_train, X_test, y_test, labels


def load_regression_dataset(
        name: str, nan_value: float = np.nan
) -> tuple[ak.Array, np.array, ak.Array, np.array]:
    root = get_project_root()
    path = root / "datasets" / "regression" / name
    X_train = ak.from_json(path / (name + "__X_train.json"))
    X_test = ak.from_json(path / (name + "__X_test.json"))
    X_train, X_test = fill_none(X_train, X_test, replace_with=nan_value)
    y_train = np.array(ak.from_json(path / (name + "__y_train.json")))
    y_test = np.array(ak.from_json(path / (name + "__y_test.json")))
    return X_train, y_train, X_test, y_test


def load_forecasting_dataset(
        name: str, nan_value: float = np.nan
) -> tuple[ak.Array, ak.Array]:
    root = get_project_root()
    path = root / "datasets" / "forecasting" / name
    X = ak.from_json(path / (name + "__X.json"))
    Y = ak.from_json(path / (name + "__Y.json"))
    X, Y = fill_none(X, Y, replace_with=nan_value)
    return X, Y


def load_dataset(name: str, nan_value: float = np.nan) -> TimeSeriesDataset:
    d = dataset_dict()
    dataset_type = None
    for task in d.keys():
        if name in d[task]:
            dataset_type = task
    assert dataset_type is not None
    if dataset_type == "classification":
        X_train, y_train, X_test, y_test, labels = load_classification_dataset(name, nan_value=nan_value)
        return TimeSeriesClassificationDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            labels=labels,
            name=name,
        )
    elif dataset_type == "regression":
        X_train, y_train, X_test, y_test = load_regression_dataset(name, nan_value=nan_value)
        return TimeSeriesRegressionDataset(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, name=name
        )
    elif dataset_type == "forecasting":
        X, Y = load_forecasting_dataset(name, nan_value=nan_value)
        return TimeSeriesForecastingDataset(X=X, Y=Y, name=name)
    else:
        raise ValueError("Dataset type not implemented.")
