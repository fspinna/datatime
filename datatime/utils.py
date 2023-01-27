import pathlib
import awkward as ak
import numpy as np
from numpy.typing import NDArray
from typing import List, Any, Dict, Tuple


def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent


def fill_none(*args: ak.Array, replace_with: float = np.nan) -> List[ak.Array]:
    return [ak.fill_none(X, replace_with) for X in args]


def get_default_dataset_path(dataset_name: str, task: str) -> pathlib.Path:
    return pathlib.Path(get_project_root()) / "cached_datasets" / task / dataset_name


def map_labels(y: NDArray[Any], labels: Dict[Any, Any]) -> Any:
    return np.vectorize(labels.get)(y)


def shape(X: ak.Array) -> Tuple:
    return len(X), ak.num(X, axis=1)[0], tuple(np.unique(ak.num(X, axis=2)))


if __name__ == "__main__":
    pass
