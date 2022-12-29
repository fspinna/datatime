import json

from datatime.gdrive_utils import (
    get_id_to_download_gdrive,
    download_file_from_google_drive,
)
from datatime.github_utils import (
    get_id_to_download_github,
    get_raw_file_dataset_url_github,
    download_raw_file_from_github,
)
from datatime.utils import get_project_root


def download_dataset(name: str, origin: str) -> None:
    if origin == "gdrive":
        to_download = get_id_to_download_gdrive(dataset_name=name)
    elif origin == "github":
        to_download = get_id_to_download_github(dataset_name=name)
    else:
        raise Exception(NotImplementedError)
    destination = (
        get_project_root()
        / "cached_datasets"
        / to_download.iloc[0]["task"]
        / to_download.iloc[0]["dataset"]
    )
    destination.mkdir(parents=True, exist_ok=True)
    if origin == "gdrive":
        for i in range(len(to_download)):
            download_file_from_google_drive(
                to_download.iloc[i]["file_id"],
                destination / to_download.iloc[i]["file"],
            )
    elif origin == "github":
        for i in range(len(to_download)):
            url = get_raw_file_dataset_url_github(
                dataset_name=name, task=to_download.iloc[i]["task"]
            )
            json_content = download_raw_file_from_github(url)
            with open(to_download.iloc[i]["file"], "w", encoding="utf8") as json_file:
                json.dump(json_content, json_file)  # FIXME: did not test if this works
    else:
        raise Exception(NotImplementedError)
