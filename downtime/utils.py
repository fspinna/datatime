import pathlib
import awkward as ak
import numpy as np
from numpy.typing import NDArray
from typing import List, Any, Dict, Tuple
from downtime.config import CACHE_FOLDER
import yaml


def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent


def fill_none(*args: ak.Array, replace_with: float = np.nan) -> List[ak.Array]:
    return [ak.fill_none(X, replace_with) for X in args]


def get_default_dataset_path(dataset_name: str) -> pathlib.Path:
    return CACHE_FOLDER / dataset_name


def map_labels(y: NDArray[Any], labels: Dict[Any, Any]) -> Any:
    return np.vectorize(labels.get)(y)


def shape(X: ak.Array) -> Tuple:
    return len(X), ak.num(X, axis=1)[0], tuple(np.unique(ak.num(X, axis=2)))


def pretty_shape(X: ak.Array) -> str:
    n, k, m = shape(X)
    return f"({n}, {k}, {m[0]})" if len(m) == 1 else f"({n}, {k}, {m[0]}:{m[-1]})"


def load_metadata(path: pathlib.Path, name: str) -> Dict:
    with open(path / (name + "__metadata.yaml")) as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)
    return metadata


if __name__ == "__main__":
    pass
