from setuptools import setup
from setuptools.config import read_configuration

if __name__ == '__main__':
    config = read_configuration('pyproject.toml')
    poetry_dependencies = config['tool']['poetry']['dependencies']

    setup(
        install_requires = [str(dep) for dep in poetry_dependencies],
    )
