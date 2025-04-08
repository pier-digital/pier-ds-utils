# Data Science Utils

A toolkit for day-to-day DS tasks such as using custom transformers or
estimators.

Found a bug or have a feature request?
[Open an issue](https://github.com/pier-digital/pier-ds-utils/issues/new/choose)!

## Usage

First, import the library:

```python
import pier_ds_utils as ds
```

### Transformers

```python
discrete_categorizer = ds.transformer.CustomDiscreteCategorizer(...)
interval_categorizer = ds.transformer.CustomIntervalCategorizer(...)
interval_categorizer_by_category = ds.transformer.CustomIntervalCategorizerByCategory(...)
log_transformer = ds.transformer.LogTransformer()
```

### Estimators

```python
glm_wrapper = ds.estimator.GLMWrapper(...)
predict_proba_selector = ds.estimator.PredictProbaSelector(...)
```

### Predictors

```python
predictor = ds.predictor.StaticGLM(...)
```

Example usage:

```python
from pier_ds_utils.predictor import StaticGLM
import pandas as pd

glm = StaticGLM(
    coefficients_map={"feature1": 0.5, "feature2": 1.5},  # required
    constant=2.0,  # optional
    os_factor=1.0,  # optional
)

df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})

# The predict is equivalent to:
# y = (0.5 * feature1 + 1.5 * feature2 + constant) * os_factor
print(glm.predict(df))  # Output: [7. 9.]
```

## Installation

```bash
pip install pier-ds-utils

# or

poetry add pier-ds-utils
```

For a specific
[version](https://github.com/pier-digital/pier-ds-utils/releases):

```bash
pip install pier-ds-utils@_version_

# or

poetry add pier-ds-utils@_version_
```

## Contributing

Contributions are welcome! Please read the
[contributing guidelines](CONTRIBUTING.md) first.
