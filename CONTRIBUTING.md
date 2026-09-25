# Contributing

## Development

### Setting up a development environment

If you don't have a local development environment, you can follow these steps to set one up.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

Now, initialize the project:

```bash
make init
```

### Running tests

You can run the tests with:

```bash
make tests
```

This will run the tests with [pytest](https://docs.pytest.org/en/latest/) and show information about the coverage.

### Formatting the code

To look for formatting issues:

```bash
make lint
```

To format the code, you can use the command:

```bash
make formatting
```

### Running all quality checks

To run the full set of quality checks (lint, code duplication, and code complexity):

```bash
make check
```

This runs [ruff](https://docs.astral.sh/ruff/) for linting/formatting, [jscpd](https://github.com/kucherenko/jscpd) for code duplication, and [complexipy](https://github.com/rohaquinlop/complexipy) for cognitive complexity. `jscpd` and `complexipy` are fetched on demand via `npx`/`uvx`, so no extra install step is needed beyond having [Node.js](https://nodejs.org/) available locally in addition to `uv`.

### Releasing a new version

To release a new version, you need to follow these steps:

1. Update the version with `uv version <version>` and commit the changes. This project follows [Semantic Versioning](http://semver.org/), so the version number should follow the format `<major>.<minor>.<patch>`.

2. Create a Github release with the new version number.

3. (Optional) Publish the new version to PyPI with `uv build && uv publish`.
