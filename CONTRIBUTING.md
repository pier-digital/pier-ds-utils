# Contributing

## Development 

### Setting up a development environment

If you don't have a local development environment, you can follow these steps to set one up.

Install [poetry](https://python-poetry.org/) and [task](https://taskfile.dev/).

Now, initialize the project:

```bash
task init
```

### Running tests

You can run the tests with:

```bash
task tests
```

This will run the tests with [pytest](https://docs.pytest.org/en/latest/) and show information about the coverage.

### Formatting the code

To look for formatting issues:
```bash
task check-formatting
```

To format the code, you can use the command:

```bash
task formatting
```

### Releasing a new version

To release a new version, you need to follow these steps:

1. Update the version with `poetry version <version>` and commit the changes. This project follows [Semantic Versioning](http://semver.org/), so the version number should follow the format `<major>.<minor>.<patch>`.

2. Create a Github release with the new version number.

3. (Optional) Publish the new version to PyPI with `poetry publish --build`.