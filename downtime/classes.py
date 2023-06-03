import awkward as ak
import pandas as pd
from numpy.typing import NDArray
from typing import Optional, Any, Dict, Tuple
from downtime.utils import map_labels, shape
from abc import ABC
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


class TimeSeriesForecastingDataset(TimeSeriesDataset):
    def __init__(
        self,
        X: Optional[ak.Array] = None,
        Y: Optional[ak.Array] = None,
        metadata: Optional[Dict] = None,
    ):
        self.X = X
        self.Y = Y
        self.XY_ = None
        self.metadata = metadata

    def __call__(
        self, *args, **kwargs
    ) -> Tuple[Optional[ak.Array], Optional[ak.Array]]:
        return self.X, self.Y

    def __str__(self) -> str:
        return "X: %s\nY: %s\nMetadata:\n%s" % (
            shape(self.X) if self.X is not None else None,
            shape(self.Y) if self.Y is not None else None,
            pprint.pformat(self.metadata) if self.metadata is not None else None,
        )

    def XY(self, cache=False) -> ak.Array:
        XY_ = self.XY_
        if XY_ is None:
            XY_ = ak.concatenate([self.X, self.Y], axis=2)
            if cache:
                self.XY_ = XY_
        return XY_


class TimeSeriesMultioutputDataset(TimeSeriesDataset):
    def __init__(
        self,
        X_train: Optional[ak.Array] = None,
        Y_train: Optional[pd.DataFrame] = None,
        X_test: Optional[ak.Array] = None,
        Y_test: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict] = None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.metadata = metadata

    def __call__(
        self, *args, **kwargs
    ) -> Tuple[
        Optional[ak.Array],
        Optional[pd.DataFrame],
        Optional[ak.Array],
        Optional[pd.DataFrame],
    ]:
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def __str__(self) -> str:
        return "X_train: %s\nX_test: %s\nY_train: %s\nY_test: %s\nMetadata:\n%s" % (
            shape(self.X_train) if self.X_train is not None else None,
            shape(self.X_test) if self.X_test is not None else None,
            self.Y_train.shape if self.Y_train is not None else None,
            self.Y_test.shape if self.Y_test is not None else None,
            self.metadata if self.metadata is not None else None,
        )
