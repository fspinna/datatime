import awkward as ak
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Optional, Any, Dict, Tuple
from downtime.utils import map_labels, shape
from abc import ABC, abstractmethod
import pprint


class TimeSeriesDataset(ABC):
    pass


class TimeSeriesClassificationDataset(TimeSeriesDataset):
    def __init__(
        self,
        X_train: Optional[ak.Array] = None,
        y_train: Optional[NDArray[Any]] = None,
        X_test: Optional[ak.Array] = None,
        y_test: Optional[NDArray[Any]] = None,
        metadata: Optional[Dict] = None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.metadata = metadata
        self.labels = None
        if metadata is not None:
            if "labels" in metadata:
                self.labels = metadata["labels"]

    def __call__(
        self, *args, **kwargs
    ) -> Tuple[
        Optional[ak.Array],
        Optional[NDArray[Any]],
        Optional[ak.Array],
        Optional[NDArray[Any]],
    ]:
        return self.X_train, self.y_train, self.X_test, self.y_test

    def __str__(self) -> str:
        return (
            "X_train: %s\nX_test: %s\ny_train: %s\ny_test: %s\nLabel Encoding: %s\nMetadata:\n%s"
            % (
                shape(self.X_train) if self.X_train is not None else None,
                shape(self.X_test) if self.X_test is not None else None,
                self.y_train.shape if self.y_train is not None else None,
                self.y_test.shape if self.y_test is not None else None,
                self.labels if self.labels is not None else None,
                pprint.pformat(self.metadata) if self.metadata is not None else None,
            )
        )

    def map_labels(self, y: NDArray[Any]) -> Any:
        return map_labels(y=y, labels=self.labels)


class TimeSeriesRegressionDataset(TimeSeriesDataset):
    def __init__(
        self,
        X_train: Optional[ak.Array] = None,
        y_train: Optional[NDArray[Any]] = None,
        X_test: Optional[ak.Array] = None,
        y_test: Optional[NDArray[Any]] = None,
        metadata: Optional[Dict] = None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.metadata = metadata

    def __call__(
        self, *args, **kwargs
    ) -> Tuple[
        Optional[ak.Array],
        Optional[NDArray[Any]],
        Optional[ak.Array],
        Optional[NDArray[Any]],
    ]:
        return self.X_train, self.y_train, self.X_test, self.y_test

    def __str__(self) -> str:
        return "X_train: %s\nX_test: %s\ny_train: %s\ny_test: %s\nMetadata:\n%s" % (
            shape(self.X_train) if self.X_train is not None else None,
            shape(self.X_test) if self.X_test is not None else None,
            self.y_train.shape if self.y_train is not None else None,
            self.y_test.shape if self.y_test is not None else None,
            pprint.pformat(self.metadata) if self.metadata is not None else None,
        )


# TODO: implement this as above
class TimeSeriesForecastingDataset(TimeSeriesDataset):
    def __init__(self, X: ak.Array, Y: ak.Array, name: str = ""):
        self.X = X
        self.Y = Y
        self.XY_ = None
        self.name = name

    def __call__(self, *args, **kwargs) -> Tuple[ak.Array, ak.Array]:
        return self.X, self.Y

    def __str__(self) -> str:
        return "Dataset Name: %s\nTask: forecasting\nX: %s\nY: %s" % (
            self.name,
            shape(self.X),
            shape(self.Y),
        )

    def XY(self) -> ak.Array:
        if self.XY_ is None:
            self.XY_ = ak.concatenate([self.X, self.Y], axis=2)
        return self.XY_


class TimeSeriesMultioutputDataset(TimeSeriesDataset):
    def __init__(
        self,
        X_train: ak.Array,
        Y_train: pd.DataFrame,
        X_test: ak.Array,
        Y_test: pd.DataFrame,
        name: str = "",
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.name = name

        self.X_ = None
        self.Y_ = None

    def __call__(
        self, *args, **kwargs
    ) -> Tuple[ak.Array, pd.DataFrame, ak.Array, pd.DataFrame]:
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def __str__(self) -> str:
        return (
            "Dataset Name: %s\nTask: multioutput\nX_train: %s\nX_test: %s\nY_train: %s\nY_test: %s"
            % (
                self.name,
                shape(self.X_train),
                shape(self.X_test),
                self.Y_train.shape,
                self.Y_test.shape,
            )
        )

    def X(self) -> ak.Array:
        if self.X_ is None:
            self.X_ = ak.concatenate([self.X_train, self.X_test], axis=0)
        return self.X_

    def Y(self) -> pd.DataFrame:
        if self.Y_ is None:
            self.Y_ = pd.concat([self.Y_train, self.Y_test], axis=0)
        return self.Y_

    def XY(self) -> Tuple[ak.Array, pd.DataFrame]:
        return self.X(), self.Y()
