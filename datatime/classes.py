import awkward as ak
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Optional, Any, Dict, Tuple
from datatime.utils import map_labels, shape


class TimeSeriesDataset:
    pass


class TimeSeriesClassificationDataset(TimeSeriesDataset):
    def __init__(
        self,
        X_train: ak.Array,
        y_train: NDArray[Any],
        X_test: ak.Array,
        y_test: NDArray[Any],
        labels: Optional[Dict[Any, Any]] = None,
        name: str = "",
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        if labels is None:
            labels = {label: str(label) for label in np.unique(y_train)}
        self.labels = labels
        self.name = name

    def __call__(
        self, *args, **kwargs
    ) -> Tuple[ak.Array, NDArray[Any], ak.Array, NDArray[Any]]:
        return self.X_train, self.y_train, self.X_test, self.y_test

    def __str__(self) -> str:
        return (
            "Dataset Name: %s\nTask: classification\nX_train: %s\nX_test: %s\ny_train: %s\ny_test: %s\nLabel "
            "Encoding: %s"
            % (
                self.name,
                shape(self.X_train),
                shape(self.X_test),
                self.y_train.shape,
                self.y_test.shape,
                self.labels,
            )
        )

    def map_labels(self, y: NDArray[Any]) -> Any:
        return map_labels(y=y, labels=self.labels)


class TimeSeriesRegressionDataset(TimeSeriesDataset):
    def __init__(
        self,
        X_train: ak.Array,
        y_train: NDArray[Any],
        X_test: ak.Array,
        y_test: NDArray[Any],
        name: str = "",
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.name = name

    def __call__(
        self, *args, **kwargs
    ) -> Tuple[ak.Array, NDArray[Any], ak.Array, NDArray[Any]]:
        return self.X_train, self.y_train, self.X_test, self.y_test

    def __str__(self) -> str:
        return (
            "Dataset Name: %s\nTask: regression\nX_train: %s\nX_test: %s\ny_train: %s\ny_test: %s"
            % (
                self.name,
                shape(self.X_train),
                shape(self.X_test),
                self.y_train.shape,
                self.y_test.shape,
            )
        )


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
