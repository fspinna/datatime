import requests

from downtime.utils import get_project_root
import pandas as pd
from downtime.config import LOCAL_DATABASE_FILENAME, DATABASE_FOLDER_NAME_COLUMN

GITHUB_DATABASE = pd.read_csv(get_project_root() / LOCAL_DATABASE_FILENAME)


def get_id_to_download_github(dataset_name: str) -> pd.DataFrame:
    df = GITHUB_DATABASE[GITHUB_DATABASE[DATABASE_FOLDER_NAME_COLUMN] == dataset_name]
    return df


def get_raw_file_dataset_url_github(
    dataset_name: str,
    task: str,
    prefix: str = "https://raw.githubusercontent.com/fspinna/datatime/main/datatime/datasets",
) -> str:
    return "{}/{}/{}".format(prefix, task, dataset_name)


def download_raw_file_from_github(url: str) -> str:
    response = requests.get(url)
    return response.text
