import requests

from datatime.utils import get_project_root
import pandas as pd

GITHUB_DATABASE = pd.read_csv(get_project_root() / "database.csv")


def get_id_to_download_github(dataset_name):
    df = GITHUB_DATABASE[GITHUB_DATABASE["dataset"] == dataset_name]
    return df


def get_raw_file_dataset_url_github(
    dataset_name,
    task,
    prefix="https://raw.githubusercontent.com/fspinna/datatime/main/datatime/datasets",
):
    return "{}/{}/{}".format(prefix, task, dataset_name)


def download_raw_file_from_github(url):
    response = requests.get(url)
    return response.text
