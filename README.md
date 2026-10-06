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

#### CustomDiscreteCategorizer

```python
discrete_categorizer = ds.transformer.CustomDiscreteCategorizer(
        column="input_col_name",
        categories=[["my_category_value_1", "my_category_value_2"], ["my_category_value_3"]],
        labels=["label_1", "label_2"],
        default_value="a-default-value",
        output_column="output_col_name",
    )
```

#### CustomIntervalCategorizer

```python
interval_categorizer = ds.transformer.CustomIntervalCategorizer(
    column="price",
    intervals=[(6700000, sys.maxsize)],
    labels=["gt_67k"],
    default_value="lt_67k",
    output_column="cat_price",
)
```

#### CustomIntervalCategorizerByCategory

```python
interval_categorizer_by_category = ds.transformer.CustomIntervalCategorizerByCategory(
    category_column: "category",
    interval_categorizers: {
        "category_1": CustomIntervalCategorizer(
            column="price",
            intervals=[(6700000, sys.maxsize)],
            labels=["gt_67k"],
            default_value="lt_67k",
            output_column="cat_price",
        ),
        "category_2": CustomIntervalCategorizer(
            column="price",
            intervals=[(0, 1000000)],
            labels=["lt_1M"],
            default_value="gt_1M",
            output_column="cat_price",
        ),
    },
    output_column = "cat_price",
)
```

#### CustomMathOperation

```python
math_operation = ds.transformer.CustomMathOperation(
    operation="multiplication",  # also accepts addition, subtraction, division
    column_a="discrete_col",
    column_b="numeric_col",
    output_column="output_col_name",
)
```

#### CustomMathOperationByConstant

```python
math_operation_by_constant = ds.transformer.CustomMathOperationByConstant(
    operation="multiplication",  # also accepts addition, subtraction, division
    column="numeric_col",
    constant=1.1,
    output_column="output_col_name",
    invert_order=False,  # set True to compute constant op column instead of column op constant
)
```

#### LogTransformer

```python
log_transformer = ds.transformer.LogTransformer()
```

#### BoundariesTransformer

```python
boundaries_transformer = ds.transformer.BoundariesTransformer(
    lower_bound=0,
    upper_bound=1000000,
)
```

```python
boundaries_transformer = ds.transformer.BoundariesTransformer(
    lower_bound=0,
    upper_bound=1000000,
    lower_value=10,
    upper_value=1200000
)
```


### Estimators

```python
glm_wrapper = ds.estimator.GLMWrapper(...)
predict_proba_selector = ds.estimator.PredictProbaSelector(...)
```

`ClusterLabelMapper` wraps a clustering estimator and translates the cluster indexes returned by `predict` into string labels:

```python
mapper = ds.estimator.ClusterLabelMapper(
    ds.kmodes.KModes(n_clusters=2, random_state=0),
    cluster_map={0: "low_risk", 1: "high_risk"},
)
mapper.fit(df).predict(df)  # array(["low_risk", "high_risk", ...], dtype=object)
```

Indexes missing from `cluster_map` raise a `ValueError`.

### Clustering

`KModes` clusters purely categorical data; `KPrototypes` clusters mixed numeric and categorical data. Categorical columns of `KPrototypes` can be given by name (DataFrame input) or by positional index. Missing values are not supported (impute them first); categories not seen during `fit` are accepted by `predict` and simply count as mismatches.

#### KModes

```python
km = ds.kmodes.KModes(n_clusters=3, init="Cao", n_init=5, random_state=0)
labels = km.fit_predict(df[["state", "channel"]])  # all columns are categorical
km.cluster_centroids_, km.cost_
```

Every column of `X` is treated as categorical, so select the columns to cluster on before calling `fit`.

| Parameter | Default | Description |
|---|---|---|
| `n_clusters` | `8` | Number of clusters (and centroids) to form. |
| `max_iter` | `100` | Maximum number of passes over the data in a single run. A run stops earlier when no point changes cluster or the cost stops improving. |
| `cat_dissim` | `None` | Dissimilarity between rows and centroids: a callable `f(points, centroids, cl_attr_freq, cl_sizes)` returning an `(n_points, n_clusters)` matrix. `None` uses `matching_dissim` (number of columns where the values differ). `ng_dissim` (frequency based, Ng et al. 2007) is available in `pier_ds_utils.kmodes.dissim`. |
| `init` | `"Cao"` | How the initial centroids are chosen. `"Huang"` samples each column from its category frequencies (random); `"Cao"` picks dense, well separated rows (deterministic); `"random"` picks random rows; or an array of shape `(n_clusters, n_columns)` with the initial centroids, in which case only one run is made. |
| `n_init` | `10` | Number of runs with different initialisations. The run with the lowest cost is kept. |
| `verbose` | `0` | When non-zero, logs the cost and number of moves of each iteration through the `logging` module. |
| `random_state` | `None` | Seed (int or `RandomState`) for reproducible results. |
| `n_jobs` | `1` | Number of parallel jobs used to run the `n_init` initialisations. `-1` uses all cores. |

#### KPrototypes

```python
kp = ds.kmodes.KPrototypes(n_clusters=3, random_state=0,
                           categorical=["state", "channel"])  # the rest is numeric
kp.fit(df)
kp.predict(df_new)
```

The distance between a row and a centroid is `numeric_dissim + gamma * categorical_dissim`. Numeric centroids are the cluster means, categorical centroids the cluster modes.

| Parameter | Default | Description |
|---|---|---|
| `n_clusters` | `8` | Number of clusters. |
| `max_iter` | `100` | Maximum number of passes over the data in a single run (see `KModes`). |
| `num_dissim` | `None` | Dissimilarity for the numeric columns, same callable signature as `cat_dissim`. `None` uses the squared euclidean distance. Scale the numeric columns first if their ranges differ a lot. |
| `cat_dissim` | `None` | Dissimilarity for the categorical columns (see `KModes`). `None` uses `matching_dissim`. |
| `gamma` | `None` | Weight of the categorical part against the numeric part: larger values make categorical mismatches matter more. `None` uses half of the mean standard deviation of the numeric columns (available after `fit` as `gamma_`). |
| `init` | `"Cao"` | Initialisation method (see `KModes`). With `"Huang"` and `"Cao"` the categorical centroids follow the chosen method and the numeric centroids are the column means plus random noise scaled by the standard deviation; `"random"` takes both parts from the same random rows. An array must have one column per column of `X`, in their original order. |
| `n_init` | `10` | Number of runs with different initialisations; the best is kept. |
| `verbose` | `0` | When non-zero, logs each iteration. |
| `random_state` | `None` | Seed for reproducible results. |
| `n_jobs` | `1` | Number of parallel jobs for the `n_init` runs. |
| `categorical` | `None` | Required. List of the categorical columns, either all names (DataFrame input) or all positional indices; names and indices cannot be mixed. Every other column is treated as numeric. It can also be passed to `fit(X, categorical=...)`, which takes precedence. |

#### Methods and attributes

- `fit(X)`, `fit_predict(X)`: fit the model (`fit_predict` also returns the labels); `predict(X)`: assign new rows to the closest cluster. `X` is a DataFrame or a 2D array.
- `labels_`: cluster of each training row. `cluster_centroids_`: array `(n_clusters, n_columns)` with the centroids in the original column order (modes for categorical, means for numeric columns).
- `cost_`: total dissimilarity of the best run (lower is better). `n_iter_`: iterations of the best run. `epoch_costs_`: cost after each iteration.
- `gamma_` (`KPrototypes`): the weight actually used. `categorical_idx_` / `numeric_idx_`: positions of the categorical and numeric columns. `feature_names_in_` / `n_features_in_`: the columns seen in `fit`, checked again in `predict`.

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

# or

uv add pier-ds-utils
```

For a specific
[version](https://github.com/pier-digital/pier-ds-utils/releases):

```bash
pip install pier-ds-utils@_version_

# or

poetry add pier-ds-utils@_version_

# or

uv add pier-ds-utils@_version_
```

## Contributing

Contributions are welcome! Please read the
[contributing guidelines](CONTRIBUTING.md) first.
