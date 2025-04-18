import pandas as pd
import pier_ds_utils as ds
from sklearn.compose import ColumnTransformer


def test_custom_discrete_categorizer():
    categorizer = ds.transformer.CustomDiscreteCategorizer(
        column="gender",
        categories=[
            ["M", "m", "Masculino", "masculino"],
            ["F", "f", "Feminino", "feminino"],
        ],
        labels=["M", "F"],
        default_value="M",
    )

    X = pd.DataFrame(
        {
            "gender": [
                "M",
                "m",
                "Masculino",
                "masculino",
                "F",
                "f",
                "Feminino",
                "feminino",
                "",
                "non-sense",
                None,
                42,
                42.42,
            ],
        }
    )

    X_transformed = categorizer.fit_transform(X)

    assert X_transformed["gender"].tolist() == [
        "M",
        "M",
        "M",
        "M",
        "F",
        "F",
        "F",
        "F",
        "M",
        "M",
        "M",
        "M",
        "M",
    ]


def test_custom_discrete_categorizer_get_params():
    categories = [
        ["M", "m", "Masculino", "masculino"],
        ["F", "f", "Feminino", "feminino"],
    ]
    labels = ["M", "F"]
    default_value = "M"

    categorizer = ds.transformer.CustomDiscreteCategorizer(
        column="gender",
        categories=categories,
        labels=labels,
        default_value=default_value,
    )

    params = categorizer.get_params()

    assert params["categories"] == categories
    assert params["labels"] == labels
    assert params["default_value"] == default_value


def test_custom_interval_categorizer():
    categorizer = ds.transformer.CustomIntervalCategorizer(
        column="price",
        intervals=[
            (498, 2700),
            (2700, 3447.6),
            (3447.6, 5592),
            (5592, 13950),
        ],
        labels=["fx1_apple", "fx2_apple", "fx3_apple", "fx4_apple"],
        default_value="fx_outras_marcas",
        output_column="price_fx",
    )

    X = pd.DataFrame(
        {
            "price": [
                498,
                2699,
                2700,
                3447.5,
                3447.6,
                5591,
                5592,
                13949,
                200,
                15999,
            ],
        }
    )

    X = categorizer.fit_transform(X)

    assert X["price_fx"].tolist() == [
        "fx1_apple",
        "fx1_apple",
        "fx2_apple",
        "fx2_apple",
        "fx3_apple",
        "fx3_apple",
        "fx4_apple",
        "fx4_apple",
        "fx_outras_marcas",
        "fx_outras_marcas",
    ]


def test_custom_interval_categorizer_get_params():
    column = "price"
    intervals = [
        (498, 2700),
        (2700, 3447.6),
        (3447.6, 5592),
        (5592, 13950),
    ]
    labels = ["fx1_apple", "fx2_apple", "fx3_apple", "fx4_apple"]
    default_value = "fx_outras_marcas"
    output_column = "price_fx"

    categorizer = ds.transformer.CustomIntervalCategorizer(
        column=column,
        intervals=intervals,
        labels=labels,
        default_value=default_value,
        output_column=output_column,
    )

    params = categorizer.get_params()

    assert params["column"] == column
    assert params["intervals"] == intervals
    assert params["labels"] == labels
    assert params["default_value"] == default_value
    assert params["output_column"] == output_column


def test_custom_interval_categorizer_by_category():
    categorizer = ds.transformer.CustomIntervalCategorizerByCategory(
        category_column="brand",
        interval_categorizers={
            "apple": ds.transformer.CustomIntervalCategorizer(
                column="price",
                intervals=[
                    (498, 2700),
                    (2700, 3447.6),
                    (3447.6, 5592),
                    (5592, 13950),
                ],
                labels=["fx1_apple", "fx2_apple", "fx3_apple", "fx4_apple"],
            ),
            "samsung": ds.transformer.CustomIntervalCategorizer(
                column="price",
                intervals=[
                    (189, 1500),
                    (1500, 11340),
                ],
                labels=["fx1_samsung", "fx2_samsung"],
            ),
        },
        default_categorizer=ds.transformer.CustomIntervalCategorizer(
            column="price",
            intervals=[(240, 5260)],
            labels=["fx_outras_marcas"],
        ),
        output_column="price_fx",
    )

    X = pd.DataFrame(
        {
            "brand": [
                "apple",
                "apple",
                "apple",
                "apple",
                "apple",
                "apple",
                "apple",
                "apple",
                "samsung",
                "samsung",
                "samsung",
                "samsung",
                "outras_marcas",
                "outras_marcas",
            ],
            "price": [
                498,
                2699,
                2700,
                3447.5,
                3447.6,
                5591,
                5592,
                13949,
                189,
                1499,
                1500,
                11339,
                240,
                5259,
            ],
        }
    )

    X = categorizer.fit_transform(X)

    assert X["price_fx"].tolist() == [
        "fx1_apple",
        "fx1_apple",
        "fx2_apple",
        "fx2_apple",
        "fx3_apple",
        "fx3_apple",
        "fx4_apple",
        "fx4_apple",
        "fx1_samsung",
        "fx1_samsung",
        "fx2_samsung",
        "fx2_samsung",
        "fx_outras_marcas",
        "fx_outras_marcas",
    ]


def test_log_transformer():
    X = pd.DataFrame(
        {
            "price": [
                498,
                2699,
                2700,
                3447.5,
                3447.6,
                5591,
                5592,
                13949,
                200,
                -15999,
            ],
        }
    )

    ct = ColumnTransformer(
        [("log", ds.transformer.LogTransformer(), ["price"])]
    ).set_output(transform="pandas")

    transformed_X = ct.fit_transform(X)
    assert transformed_X is not None
    assert transformed_X.shape == (10, 1)
    assert transformed_X.columns.tolist() == ["log__price"]


def test_log_transformer_with_multiple_columns():
    X = pd.DataFrame(
        {
            "price": [
                498,
                2699,
                2700,
                3447.5,
                3447.6,
                5591,
                5592,
                13949,
                200,
                -15999,
            ],
            "price2": [
                498,
                2699,
                2700,
                3447.5,
                3447.6,
                5591,
                5592,
                13949,
                200,
                -15999,
            ],
            "category": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
            ],
        }
    )

    ct = ColumnTransformer(
        [("log", ds.transformer.LogTransformer(), ["price", "price2"])]
    ).set_output(transform="pandas")

    transformed_X = ct.fit_transform(X)
    assert transformed_X is not None
    assert transformed_X.shape == (10, 2)
    assert transformed_X.columns.tolist() == ["log__price", "log__price2"]


def test_boundaries_transformer():
    X = pd.DataFrame(
        {
            "price": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    ct = ColumnTransformer(
        [
            (
                "boundaries",
                ds.transformer.BoundariesTransformer(lower_bound=2, upper_bound=9),
                ["price"],
            )
        ]
    ).set_output(transform="pandas")

    transformed_X = ct.fit_transform(X)
    assert transformed_X is not None
    assert transformed_X.shape == (10, 1)
    assert transformed_X.columns.tolist() == ["boundaries__price"]
    assert transformed_X["boundaries__price"].tolist() == [2, 2, 3, 4, 5, 6, 7, 8, 9, 9]
