# source: https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
import requests
import pandas as pd
from datatime.utils import get_project_root
from tqdm import tqdm


CHUNK_SIZE = 32768
URL = "https://docs.google.com/uc?export=download&confirm"
GDRIVE_DATABASE = pd.read_csv(get_project_root() / "gdrive_database.csv")


def download_file_from_google_drive(id: str, destination: str) -> None:
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_id_to_download_gdrive(dataset_name: str) -> pd.DataFrame:
    df = GDRIVE_DATABASE[GDRIVE_DATABASE["dataset"] == dataset_name]
    return df
