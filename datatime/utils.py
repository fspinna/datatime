import pathlib
import awkward as ak
import numpy as np

from datatime.database_utils import datasets_df


def get_project_root():
    return pathlib.Path(__file__).parent


def fill_none(*args: ak.Array, replace_with: float = np.nan) -> list[ak.Array]:
    return [ak.fill_none(X, replace_with) for X in args]


def get_default_dataset_path(dataset_name, task):
    return pathlib.Path(get_project_root()) / "cached_datasets" / task / dataset_name


if __name__ == "__main__":
    df = datasets_df()