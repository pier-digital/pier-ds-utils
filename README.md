# Data Science Utils

A toolkit for day-to-day DS tasks such as using custom transformers or estimators.

Found a bug or have a feature request? [Open an issue](https://github.com/pier-digital/pier-ds-utils/issues/new/choose)!

## Usage

```python
import pier_ds_utils as ds

# Transformers
discrete_categorizer = ds.transformer.CustomDiscreteCategorizer(...)
interval_categorizer = ds.transformer.CustomIntervalCategorizer(...)
interval_categorizer_by_category = ds.transformer.CustomIntervalCategorizerByCategory(...)

# Estimators
glm_wrapper = ds.estimator.GLMWrapper(...)
predict_proba_selector = ds.estimator.PredictProbaSelector(...)
```

## Installation

```bash
pip install pier-ds-utils

# or

poetry add pier-ds-utils
```

For a specific [version](https://github.com/pier-digital/pier-ds-utils/releases):
```bash
pip install pier-ds-utils@_version_

# or

poetry add pier-ds-utils@_version_
```

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first.