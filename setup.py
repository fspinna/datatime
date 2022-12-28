from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="datatime",
    version="",
    packages=["datatime"],
    url="",
    license="",
    author="francesco",
    author_email="",
    description="",
    install_requires=[requirements],
    package_data={"datatime": ["database.csv", "gdrive_database.csv"]}
)
