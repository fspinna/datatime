import yaml
from typing import Union, Any, Optional, List, Tuple, Dict
from numpy.typing import NDArray
import awkward as ak
import numpy as np
import pandas as pd
import pathlib

from downtime.config import (
    CACHE_FOLDER,
    LOCAL_DATABASE_FILENAME,
    DATABASE_FOLDER_NAME_COLUMN,
    DATABASE_TASK_COLUMN,
    DATABASE_FILENAME_COLUMN,
    DATABASE_FILESIZE_COLUMN,
    DATABASE_DATASET_NAME_COLUMN,
    METADATA_TASK_KEY,
)
from downtime.download_utils import download_dataset
from downtime.classes import (
    TimeSeriesClassificationDataset,
    TimeSeriesRegressionDataset,
    TimeSeriesForecastingDataset,
    TimeSeriesMultioutputDataset,
)
from downtime.utils import (
    get_project_root,
    get_default_dataset_path,
    fill_none,
    load_metadata,
)


def datasets_table(tasks: Optional[List[str]] = None) -> pd.DataFrame:
    df = (
        pd.read_csv(get_project_root() / LOCAL_DATABASE_FILENAME)
        .drop(
            [
                DATABASE_FILENAME_COLUMN,
                DATABASE_FILESIZE_COLUMN,
                DATABASE_DATASET_NAME_COLUMN,
            ],
            axis=1,
        )
        .drop_duplicates()
    )
    if tasks is None:
        return df.sort_values([DATABASE_FOLDER_NAME_COLUMN])
    else:
        return df[df[DATABASE_TASK_COLUMN].isin(tasks)].sort_values(
            [DATABASE_FOLDER_NAME_COLUMN]
        )


def datasets_list(tasks: Optional[List[str]] = None) -> List[str]:
    return datasets_table(tasks=tasks)[DATABASE_FOLDER_NAME_COLUMN].to_list()


def cached_datasets_dict(root: Optional[pathlib.Path] = None) -> Dict[str, List[str]]:
    if root is None:
        root = CACHE_FOLDER
    root.mkdir(parents=True, exist_ok=True)
    out = dict()
    folders = sorted([d for d in root.iterdir() if d.is_dir()])
    for folder in folders:
        dataset = folder.name
        with open(folder / (dataset + "__metadata.yaml"), "r") as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)
        task = metadata[METADATA_TASK_KEY]
        out[dataset] = task
    return out


def cached_datasets_list(root: Optional[pathlib.Path] = None) -> List[str]:
    if root is None:
        root = CACHE_FOLDER
    root.mkdir(parents=True, exist_ok=True)
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def is_cached(dataset_name: str) -> bool:
    if dataset_name in cached_datasets_list():
        return True
    else:
        return False


def load_dataset(
    name: str,
    nan_value: float = np.nan,
    origin: str = "gdrive",
    path: Optional[pathlib.Path] = None,
) -> Union[
    TimeSeriesClassificationDataset,
    TimeSeriesRegressionDataset,
    TimeSeriesForecastingDataset,
    TimeSeriesMultioutputDataset,
]:
    # create docstring from the function signature
    """
    Load a time series dataset.
    :param name: The name of the dataset to load.
    :param nan_value: The value that represents a missing value.
    :param origin: The origin of the dataset.
    :param path: The path to the dataset.
    :return: A TimeSeriesDataset object.
    """
    d = datasets_table()
    dataset_type = d[d[DATABASE_FOLDER_NAME_COLUMN] == name][
        DATABASE_TASK_COLUMN
    ].values[0]
    if dataset_type == "classification":
        X_train, y_train, X_test, y_test, metadata = load_classification_dataset(
            name, nan_value=nan_value, origin=origin, path=path
        )
        return TimeSeriesClassificationDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            metadata=metadata,
        )
    elif dataset_type == "regression":
        X_train, y_train, X_test, y_test, metadata = load_regression_dataset(
            name, nan_value=nan_value, origin=origin, path=path
        )
        return TimeSeriesRegressionDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            metadata=metadata,
        )
    elif dataset_type == "forecasting":
        X, Y, metadata = load_forecasting_dataset(
            name, nan_value=nan_value, origin=origin, path=path
        )
        return TimeSeriesForecastingDataset(X=X, Y=Y, metadata=metadata)
    elif dataset_type == "multioutput":
        X_train, Y_train, X_test, Y_test, metadata = load_multioutput_dataset(
            name, nan_value=nan_value, origin=origin, path=path
        )
        return TimeSeriesMultioutputDataset(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            metadata=metadata,
        )
    else:
        raise ValueError("Dataset type not implemented.")


def get_path(
    name: str, path: Optional[str] = None, origin: str = "gdrive"
) -> pathlib.Path:
    if path is None:
        path = get_default_dataset_path(dataset_name=name)
        if not is_cached(dataset_name=name):
            download_dataset(name=name, origin=origin)
    else:
        path = pathlib.Path(path)
    return path


def load_classification_dataset(
    name: str,
    nan_value: float = np.nan,
    origin: str = "gdrive",
    path: Optional[str] = None,
) -> Tuple[ak.Array, NDArray[Any], ak.Array, NDArray[Any], Dict]:
    path = get_path(name=name, path=path, origin=origin)
    X_train = ak.from_json(path / (name + "__X_train.json"))
    X_test = ak.from_json(path / (name + "__X_test.json"))
    X_train, X_test = fill_none(X_train, X_test, replace_with=nan_value)
    y_train = np.array(ak.from_json(path / (name + "__y_train.json")))
    y_test = np.array(ak.from_json(path / (name + "__y_test.json")))
    metadata = load_metadata(path=path, name=name)
    return X_train, y_train, X_test, y_test, metadata


def load_regression_dataset(
    name: str,
    nan_value: float = np.nan,
    origin: str = "gdrive",
    path: Optional[str] = None,
) -> Tuple[ak.Array, NDArray[Any], ak.Array, NDArray[Any], Dict]:
    return load_classification_dataset(
        name=name, nan_value=nan_value, origin=origin, path=path
    )


def load_forecasting_dataset(
    name: str,
    nan_value: float = np.nan,
    origin: str = "gdrive",
    path: Optional[str] = None,
) -> Tuple[ak.Array, ak.Array, Dict]:
    path = get_path(name=name, path=path, origin=origin)
    X = ak.from_json(path / (name + "__X.json"))
    Y = ak.from_json(path / (name + "__Y.json"))
    X, Y = fill_none(X, Y, replace_with=nan_value)
    metadata = load_metadata(path=path, name=name)
    return X, Y, metadata


def load_multioutput_dataset(
    name: str,
    nan_value: float = np.nan,
    origin="gdrive",
    path: Optional[str] = None,
    load_train: bool = True,
    load_test: bool = True,
) -> Tuple[
    Union[ak.Array, None], pd.DataFrame, Union[ak.Array, None], pd.DataFrame, Dict
]:
    path = get_path(name=name, path=path, origin=origin)
    metadata = load_metadata(path=path, name=name)
    if load_train:
        X_train = ak.from_json(path / (name + "__X_train.json"))
        X_train = fill_none(X_train, replace_with=nan_value)[0]
    else:
        X_train = None
    if load_test:
        X_test = ak.from_json(path / (name + "__X_test.json"))
        X_test = fill_none(X_test, replace_with=nan_value)[0]
    else:
        X_test = None

    Y_train = pd.read_csv(path / (name + "__Y_train.csv"))
    Y_test = pd.read_csv(path / (name + "__Y_test.csv"))
    return X_train, Y_train, X_test, Y_test, metadata
