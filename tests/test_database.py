import pytest
from downtime.utils import get_project_root
from downtime.config import (
    LOCAL_DATABASE_FILENAME,
    GDRIVE_DATABASE_FILENAME,
    GDRIVE_DATABASE_FILE_ID_COLUMN,
    DATABASE_DATASET_NAME_COLUMN,
    DATABASE_TASK_COLUMN,
    DATABASE_FILESIZE_COLUMN,
)
import pandas as pd


@pytest.fixture
def databases():
    database = pd.read_csv(get_project_root() / LOCAL_DATABASE_FILENAME)
    database_gdrive = pd.read_csv(get_project_root() / GDRIVE_DATABASE_FILENAME)
    return database, database_gdrive


def test_database_match(databases):
    database, database_gdrive = databases
    database.drop(
        [DATABASE_DATASET_NAME_COLUMN, DATABASE_TASK_COLUMN, DATABASE_FILESIZE_COLUMN],
        axis=1,
        inplace=True,
    )
    assert database.equals(
        database_gdrive.drop([GDRIVE_DATABASE_FILE_ID_COLUMN], axis=1)
    )


def test_no_duplicates(databases):
    database, database_gdrive = databases
    assert len(database[database.duplicated()]) == 0
    assert (
        len(
            database_gdrive[
                database_gdrive.drop(
                    [GDRIVE_DATABASE_FILE_ID_COLUMN], axis=1
                ).duplicated()
            ]
        )
        == 0
    )


if __name__ == "__main__":
    pytest.main()
