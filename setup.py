from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open(path.join(here, "requirements_dev.txt"), encoding="utf-8") as f:
    requirements_dev = f.read().splitlines()

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
    extras_require={"dev": [requirements_dev]},
    package_data={"datatime": ["database.csv", "gdrive_database.csv"]}
)
