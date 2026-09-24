import pandas as pd
import pier_ds_utils as ds
import pytest
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


def test_custom_math_operation_multiplication():
    operation = ds.transformer.CustomMathOperation(
        operation="multiplication",
        column_a="discrete_col",
        column_b="numeric_col",
        output_column="output_col_name",
    )

    X = pd.DataFrame(
        {
            "discrete_col": [1, 2, 3, 4],
            "numeric_col": [10, 20, 30, 40],
        }
    )

    X = operation.fit_transform(X)

    assert X["output_col_name"].tolist() == [10, 40, 90, 160]


def test_custom_math_operation_addition():
    operation = ds.transformer.CustomMathOperation(
        operation="addition",
        column_a="a",
        column_b="b",
        output_column="sum",
    )

    X = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

    X = operation.fit_transform(X)

    assert X["sum"].tolist() == [11, 22, 33]


def test_custom_math_operation_subtraction():
    operation = ds.transformer.CustomMathOperation(
        operation="subtraction",
        column_a="a",
        column_b="b",
        output_column="diff",
    )

    X = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]})

    X = operation.fit_transform(X)

    assert X["diff"].tolist() == [9, 18, 27]


def test_custom_math_operation_division():
    operation = ds.transformer.CustomMathOperation(
        operation="division",
        column_a="a",
        column_b="b",
        output_column="ratio",
    )

    X = pd.DataFrame({"a": [10, 20, 30], "b": [2, 5, 3]})

    X = operation.fit_transform(X)

    assert X["ratio"].tolist() == [5, 4, 10]


def test_custom_math_operation_invalid_operation():
    with pytest.raises(ValueError):
        ds.transformer.CustomMathOperation(
            operation="not-a-valid-operation",
            column_a="a",
            column_b="b",
            output_column="output",
        )


def test_custom_math_operation_requires_output_column():
    with pytest.raises(TypeError):
        ds.transformer.CustomMathOperation(
            operation="multiplication",
            column_a="a",
            column_b="b",
        )


def test_custom_math_operation_get_params():
    operation = ds.transformer.CustomMathOperation(
        operation="multiplication",
        column_a="a",
        column_b="b",
        output_column="output",
    )

    params = operation.get_params()

    assert params["operation"] == "multiplication"
    assert params["column_a"] == "a"
    assert params["column_b"] == "b"
    assert params["output_column"] == "output"


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


def test_boundaries_transformer_with_custom_values():
    X = pd.DataFrame(
        {
            "price": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    ct = ColumnTransformer(
        [
            (
                "boundaries",
                ds.transformer.BoundariesTransformer(
                    lower_bound=2,
                    upper_bound=9,
                    lower_value=0,
                    upper_value=99,
                ),
                ["price"],
            )
        ]
    ).set_output(transform="pandas")

    transformed_X = ct.fit_transform(X)

    assert transformed_X is not None
    assert transformed_X.shape == (10, 1)
    assert transformed_X.columns.tolist() == ["boundaries__price"]
    assert transformed_X["boundaries__price"].tolist() == [
        0,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        99,
    ]
