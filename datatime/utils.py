import pathlib
import awkward as ak
import numpy as np
from typing import List


def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent


def fill_none(*args: ak.Array, replace_with: float = np.nan) -> List[ak.Array]:
    return [ak.fill_none(X, replace_with) for X in args]


def get_default_dataset_path(dataset_name: str, task: str) -> pathlib.Path:
    return pathlib.Path(get_project_root()) / "cached_datasets" / task / dataset_name
