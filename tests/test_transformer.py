import numpy as np
import pandas as pd
import pier_ds_utils as ds
import pytest
from sklearn.compose import ColumnTransformer


def test_custom_discrete_categorizer_output_type():
    X = pd.DataFrame({"gender": ["M", "F", "x"]})
    default = ds.transformer.CustomDiscreteCategorizer(
        column="gender",
        categories=[["M"]],
        labels=[1],
        default_value=0,
        output_column="d",
    )
    assert default.output_type_ == "object"
    assert default.fit_transform(X.copy())["d"].dtype == object

    typed = ds.transformer.CustomDiscreteCategorizer(
        column="gender",
        categories=[["M"]],
        labels=[1],
        default_value=0,
        output_column="d",
        output_type="float64",
    )
    assert typed.get_params()["output_type"] == "float64"
    out = typed.fit_transform(X.copy())["d"]
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, 0.0, 0.0]


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
    assert params["output_type"] == "object"


def test_custom_interval_categorizer_output_type():
    X = pd.DataFrame({"price": [1, 5, 20]})

    default = ds.transformer.CustomIntervalCategorizer(
        column="price", intervals=[(0, 10)], labels=[1.0], default_value=2.0
    ).fit_transform(X.copy())
    assert default["price"].dtype == object

    as_float = ds.transformer.CustomIntervalCategorizer(
        column="price",
        intervals=[(0, 10)],
        labels=[1],
        default_value=2,
        output_type="float",
    ).fit_transform(X.copy())
    assert as_float["price"].dtype == "float64"
    assert as_float["price"].tolist() == [1.0, 1.0, 2.0]

    as_category = ds.transformer.CustomIntervalCategorizer(
        column="price",
        intervals=[(0, 10)],
        labels=["low"],
        default_value="high",
        output_type="category",
    ).fit_transform(X.copy())
    assert as_category["price"].dtype == "category"


def test_custom_interval_categorizer_by_category_output_type():
    def build(**kwargs):
        return ds.transformer.CustomIntervalCategorizerByCategory(
            category_column="brand",
            interval_categorizers={
                "apple": ds.transformer.CustomIntervalCategorizer(
                    column="price", intervals=[(0, 10)], labels=[1]
                ),
            },
            default_categorizer=ds.transformer.CustomIntervalCategorizer(
                column="price", intervals=[(0, 10)], labels=[2]
            ),
            output_column="out",
            **kwargs,
        )

    X = pd.DataFrame({"brand": ["apple", "other"], "price": [5, 5]})

    default = build()
    assert default.output_type_ == "object"
    assert default.fit_transform(X.copy())["out"].dtype == object

    typed = build(output_type="float64")
    assert typed.get_params()["output_type"] == "float64"
    out = typed.fit_transform(X.copy())["out"]
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, 2.0]


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


def test_custom_math_operation_exponentiation():
    operation = ds.transformer.CustomMathOperation(
        operation="exponentiation",
        column_a="a",
        column_b="b",
        output_column="power",
    )

    X = pd.DataFrame({"a": [1, 2, 3], "b": [2, 2, 2]})

    X = operation.fit_transform(X)

    assert X["power"].tolist() == [1, 4, 9]


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


def test_custom_math_operation_by_constant_multiplication():
    operation = ds.transformer.CustomMathOperationByConstant(
        operation="multiplication",
        column="numeric_col",
        constant=10,
        output_column="output_col_name",
    )

    X = pd.DataFrame({"numeric_col": [1, 2, 3, 4]})

    X = operation.fit_transform(X)

    assert X["output_col_name"].tolist() == [10, 20, 30, 40]


def test_custom_math_operation_by_constant_addition():
    operation = ds.transformer.CustomMathOperationByConstant(
        operation="addition",
        column="a",
        constant=10,
        output_column="sum",
    )

    X = pd.DataFrame({"a": [1, 2, 3]})

    X = operation.fit_transform(X)

    assert X["sum"].tolist() == [11, 12, 13]


def test_custom_math_operation_by_constant_subtraction():
    operation = ds.transformer.CustomMathOperationByConstant(
        operation="subtraction",
        column="a",
        constant=1,
        output_column="diff",
    )

    X = pd.DataFrame({"a": [10, 20, 30]})

    X = operation.fit_transform(X)

    assert X["diff"].tolist() == [9, 19, 29]


def test_custom_math_operation_by_constant_division():
    operation = ds.transformer.CustomMathOperationByConstant(
        operation="division",
        column="a",
        constant=2,
        output_column="ratio",
    )

    X = pd.DataFrame({"a": [10, 20, 30]})

    X = operation.fit_transform(X)

    assert X["ratio"].tolist() == [5, 10, 15]


def test_custom_math_operation_by_constant_invalid_operation():
    with pytest.raises(ValueError):
        ds.transformer.CustomMathOperationByConstant(
            operation="not-a-valid-operation",
            column="a",
            constant=1,
            output_column="output",
        )


def test_custom_math_operation_by_constant_requires_output_column():
    with pytest.raises(TypeError):
        ds.transformer.CustomMathOperationByConstant(
            operation="multiplication",
            column="a",
            constant=1,
        )


def test_custom_math_operation_by_constant_get_params():
    operation = ds.transformer.CustomMathOperationByConstant(
        operation="multiplication",
        column="a",
        constant=10,
        output_column="output",
    )

    params = operation.get_params()

    assert params["operation"] == "multiplication"
    assert params["column"] == "a"
    assert params["constant"] == 10
    assert params["output_column"] == "output"
    assert params["invert_order"] is False


def test_custom_math_operation_by_constant_subtraction_inverted():
    operation = ds.transformer.CustomMathOperationByConstant(
        operation="subtraction",
        column="a",
        constant=100,
        output_column="diff",
        invert_order=True,
    )

    X = pd.DataFrame({"a": [10, 20, 30]})

    X = operation.fit_transform(X)

    assert X["diff"].tolist() == [90, 80, 70]


def test_custom_math_operation_by_constant_division_inverted():
    operation = ds.transformer.CustomMathOperationByConstant(
        operation="division",
        column="a",
        constant=100,
        output_column="ratio",
        invert_order=True,
    )

    X = pd.DataFrame({"a": [10, 20, 25]})

    X = operation.fit_transform(X)

    assert X["ratio"].tolist() == [10, 5, 4]


def test_custom_math_operation_by_constant_exponentiation():
    operation = ds.transformer.CustomMathOperationByConstant(
        operation="exponentiation",
        column="a",
        constant=2,
        output_column="power",
    )

    X = pd.DataFrame({"a": [1, 2, 3]})

    X = operation.fit_transform(X)

    assert X["power"].tolist() == [1, 4, 9]


def test_custom_math_operation_by_constant_exponentiation_inverted():
    operation = ds.transformer.CustomMathOperationByConstant(
        operation="exponentiation",
        column="a",
        constant=2,
        output_column="power",
        invert_order=True,
    )

    X = pd.DataFrame({"a": [1, 2, 3]})

    X = operation.fit_transform(X)

    assert X["power"].tolist() == [2, 4, 8]


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


def test_custom_binarizer_default_matches_sklearn():
    from sklearn.preprocessing import Binarizer

    X = pd.DataFrame({"a": [-1.0, 0.0, 1.0, 2.0], "b": [3.0, 0.0, -2.0, 0.5]})

    result = ds.transformer.CustomBinarizer().fit_transform(X)

    expected = Binarizer(threshold=0.0).fit_transform(X)
    assert result.to_numpy().tolist() == expected.tolist()


@pytest.mark.parametrize(
    "condition, expected",
    [
        (">", [0, 0, 1]),
        (">=", [0, 1, 1]),
        ("=", [0, 1, 0]),
        ("<", [1, 0, 0]),
        ("<=", [1, 1, 0]),
    ],
)
def test_custom_binarizer_conditions(condition, expected):
    X = pd.DataFrame({"a": [1, 5, 9]})

    result = ds.transformer.CustomBinarizer(
        threshold=5, condition=condition
    ).fit_transform(X)

    assert result["a"].tolist() == expected


def test_custom_binarizer_custom_values():
    X = pd.DataFrame({"a": [1, 5, 9]})

    binarizer = ds.transformer.CustomBinarizer(
        threshold=5, condition=">=", true_value="high", false_value="low"
    )

    assert binarizer.fit_transform(X)["a"].tolist() == ["low", "high", "high"]


def test_custom_binarizer_invalid_condition():
    with pytest.raises(ValueError):
        ds.transformer.CustomBinarizer(condition="!=")


def test_custom_binarizer_get_params():
    binarizer = ds.transformer.CustomBinarizer(
        threshold=3, condition="<=", true_value=10, false_value=-1
    )

    assert binarizer.get_params() == {
        "threshold": 3,
        "condition": "<=",
        "true_value": 10,
        "false_value": -1,
    }


def test_custom_binarizer_nan_gets_false_value():
    X = pd.DataFrame({"a": [1.0, np.nan, 9.0]})

    result = ds.transformer.CustomBinarizer(threshold=5, condition="<").fit_transform(X)

    assert result["a"].tolist() == [1, 0, 0]


def test_custom_binarizer_preserves_index_columns_and_input():
    X = pd.DataFrame({"a": [1, 9], "b": [9, 1]}, index=[10, 20])
    original = X.copy()

    result = ds.transformer.CustomBinarizer(threshold=5).fit_transform(X)

    assert result.index.tolist() == [10, 20]
    assert result.columns.tolist() == ["a", "b"]
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize(
    "direction, decimals, values, expected",
    [
        ("up", 0, [1.2, -1.2, 2.0], [2.0, -1.0, 2.0]),
        ("down", 0, [1.2, -1.2, 2.0], [1.0, -2.0, 2.0]),
        ("nearest", 0, [1.2, -1.2, 1.7], [1.0, -1.0, 2.0]),
        ("nearest", 0, [0.5, 1.5, 2.5], [0.0, 2.0, 2.0]),
        ("up", 1, [1.11, 1.1, 1.19], [1.2, 1.1, 1.2]),
        ("down", 2, [1.119, 1.1], [1.11, 1.1]),
        ("up", -2, [1234.5], [1300.0]),
        ("down", -2, [1234.5], [1200.0]),
    ],
)
def test_custom_rounder(direction, decimals, values, expected):
    rounder = ds.transformer.CustomRounder(direction=direction, decimals=decimals)
    X = rounder.fit_transform(pd.DataFrame({"x": values}))
    assert X["x"].tolist() == pytest.approx(expected)


def test_custom_rounder_multiple_columns_nan_and_input_untouched():
    X = pd.DataFrame({"a": [1.2, np.nan], "b": [2.01, -0.5]}, index=["i", "j"])
    original = X.copy()

    result = ds.transformer.CustomRounder().fit_transform(X)

    assert result.index.tolist() == ["i", "j"]
    assert result.columns.tolist() == ["a", "b"]
    assert result["a"].tolist()[0] == 2.0
    assert np.isnan(result["a"].tolist()[1])
    assert result["b"].tolist() == [3.0, -0.0]
    pd.testing.assert_frame_equal(X, original)


def test_custom_rounder_invalid_params():
    with pytest.raises(ValueError):
        ds.transformer.CustomRounder(direction="sideways")
    with pytest.raises(TypeError):
        ds.transformer.CustomRounder(decimals=1.5)
    with pytest.raises(TypeError):
        ds.transformer.CustomRounder(decimals=True)


def test_custom_rounder_get_params():
    rounder = ds.transformer.CustomRounder(direction="down", decimals=2)
    assert rounder.get_params() == {"direction": "down", "decimals": 2}
