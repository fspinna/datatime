import json

import awkward as ak
import numpy as np

from datatime.download_utils import download_dataset
from datatime.utils import get_default_dataset_path, fill_none
from datatime.database_utils import is_cached


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
