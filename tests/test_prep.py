from pier_ds_utils.prep import add_constant
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "input_df, prepend, column_name, constant_value, expected_df",
    [
        (
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            True,
            "const",
            1.0,
            pd.DataFrame({"const": [1.0, 1.0, 1.0], "A": [1, 2, 3], "B": [4, 5, 6]}),
        ),
        (
            pd.DataFrame({"A": [1, 2, 3], "another_column": [4, 5, 6]}),
            True,
            "another_column",
            3,
            pd.DataFrame({"another_column": [3, 3, 3], "A": [1, 2, 3]}),
        ),
        (
            pd.DataFrame({"A": [1, 2, 3], "another_column": [4, 5, 6]}),
            False,
            "another_column",
            -1,
            pd.DataFrame({"A": [1, 2, 3], "another_column": [-1, -1, -1]}),
        ),
        (
            pd.DataFrame([{"A": 1, "B": 2}]),
            True,
            "const",
            1.0,
            pd.DataFrame({"const": [1.0], "A": [1], "B": [2]}),
        ),
    ],
)
def test_add_constant_column(
    input_df, prepend, column_name, constant_value, expected_df
):
    df_with_const = add_constant(
        input_df,
        prepend=prepend,
        column_name=column_name,
        constant_value=constant_value,
    )
    assert df_with_const.equals(expected_df)
