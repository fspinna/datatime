import pathlib

CACHE_FOLDER = pathlib.Path.home() / ".downtime_cache"

GDRIVE_DATABASE_FILENAME = "database_gdrive.csv"
GDRIVE_DATABASE_FILE_ID_COLUMN = "file_id"

LOCAL_DATABASE_FILENAME = "database_local.csv"
DATABASE_FOLDER_NAME_COLUMN = "folder_name"
DATABASE_DATASET_NAME_COLUMN = "dataset_name"
DATABASE_TASK_COLUMN = "task"
DATABASE_FILENAME_COLUMN = "file"
DATABASE_FILESIZE_COLUMN = "filesize"

METADATA_LABELS_KEY = "labels"
METADATA_TASK_KEY = "task"
