from setuptools import setup, find_packages
from setuptools.config import read_configuration

if __name__ == "__main__":
    config = read_configuration("pyproject.toml")
    poetry_dependencies = config["tool"]["poetry"]["dependencies"]

    setup(
        packages=find_packages(),
        install_requires=[str(dep) for dep in poetry_dependencies],
    )
