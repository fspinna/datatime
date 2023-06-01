import awkward as ak
from numpy.typing import NDArray
from typing import Self, Callable


class Wrapper:
    def __init__(self, model, conversion_function: Callable):
        self.model = model
        self.conversion_function = conversion_function

    def fit(self, X: ak.Array, y: NDArray) -> Self:
        self.model.fit(self.conversion_function(X), y)
        return self

    def predict(self, X: ak.Array) -> NDArray:
        return self.model.predict(self.conversion_function(X))

    def predict_proba(self, X: ak.Array) -> NDArray:
        return self.model.predict_proba(self.conversion_function(X))

    def transform(self, X: ak.Array) -> NDArray:
        return self.model.transform(self.conversion_function(X))
