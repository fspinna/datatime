import awkward as ak
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Any, Dict, Tuple


class TimeSeriesDataset(object):
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

    def map_labels(self, y: NDArray[Any]) -> Any:
        return np.vectorize(self.labels.get)(y)


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

    def __call__(self, *args, **kwargs) -> Tuple[ak.Array, ak.Array]:
        return self.X, self.Y

    def __str__(self) -> str:
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

    def _concatenate_X_Y(self) -> ak.Array:
        return ak.concatenate([self.X, self.Y], axis=2)
