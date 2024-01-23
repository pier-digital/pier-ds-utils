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
# pip
pip install git+https://github.com/pier-digital/pier-ds-utils@_version_

# poetry
poetry add git+https://github.com/pier-digital/pier-ds-utils@_version_

# conda (must be in the virtual environment)
pip install git+https://github.com/pier-digital/pier-ds-utils@_version_
```

Replace `_version_` by the specific version you want to use. You can find them [here](https://github.com/pier-digital/pier-ds-utils/tags).
