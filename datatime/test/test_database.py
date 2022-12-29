import pytest
from datatime.utils import get_project_root
import pandas as pd


@pytest.fixture
def databases():
    database = pd.read_csv(get_project_root() / "database.csv")
    database_gdrive = pd.read_csv(get_project_root() / "gdrive_database.csv")
    return database, database_gdrive


def test_database_match(databases):
    database, database_gdrive = databases
    assert database.equals(database_gdrive.drop(["file_id"], axis=1))


def test_no_duplicates(databases):
    database, database_gdrive = databases
    assert len(database[database.duplicated()]) == 0
    assert (
        len(database_gdrive[database_gdrive.drop(["file_id"], axis=1).duplicated()])
        == 0
    )


if __name__ == "__main__":
    pytest.main()
