import unittest
from datatime.utils import get_project_root
import pandas as pd


class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database = pd.read_csv(get_project_root() / "database.csv")
        cls.database_gdrive = pd.read_csv(get_project_root() / "gdrive_database.csv")

    def test_database_match(self):
        self.assertTrue(
            self.database.equals(self.database_gdrive.drop(["file_id"], axis=1))
        )

    def test_no_duplicates(self):
        self.assertTrue(len(self.database[self.database.duplicated()]) == 0)
        self.assertTrue(
            len(
                self.database_gdrive[
                    self.database_gdrive.drop(["file_id"], axis=1).duplicated()
                ]
            )
            == 0
        )


if __name__ == "__main__":
    unittest.main()
