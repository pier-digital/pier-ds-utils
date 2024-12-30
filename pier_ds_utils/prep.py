import pandas as pd
import typing


def add_constant(
    df: typing.Union[pd.DataFrame, pd.Series],
    prepend: bool = True,
    column_name: str = "const",
    constant_value: typing.Any = 1.0,
) -> pd.DataFrame:
    """
    Add a constant column to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to which the constant column will be added.
    prepend : bool, default=True
        If True, the constant column will be added as the first column.
        If False, the constant column will be added as the last column.
    column_name : str, default="const"
        The name of the constant column.
    constant_value : Any, default=1.0
        The value to be used for the constant column.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the constant column added.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a DataFrame or Series")

    df = df.copy()

    original_column_names = df.columns.tolist()

    df[column_name] = constant_value

    if column_name in original_column_names:
        original_column_names.remove(column_name)

    column_names = (
        [column_name] + original_column_names
        if prepend
        else original_column_names + [column_name]
    )

    return df[column_names]
