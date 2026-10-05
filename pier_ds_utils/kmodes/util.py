"""Helpers shared by the K-Modes / K-Prototypes estimators.
"""

import numbers
import typing

import numpy as np
import pandas as pd


def resolve_columns(
    spec: typing.Sequence[typing.Union[str, int]],
    n_features: int,
    columns: typing.Optional[typing.Sequence] = None,
) -> typing.List[int]:
    """Translate a list of column names or positional indices into sorted indices.

    Parameters
    ----------
    spec: sequence of str or int
        Column names (``str``) or positional indices (``int``). Mixing both is
        not allowed.
    n_features: int
        Number of columns of the data.
    columns: sequence, optional
        Column names of the data. Required when ``spec`` holds names.
    """
    spec = list(spec)
    if not spec:
        raise ValueError("The list of categorical columns must not be empty.")
    if len(set(spec)) != len(spec):
        raise ValueError("The list of categorical columns has duplicates.")

    is_int = [isinstance(s, numbers.Integral) and not isinstance(s, bool) for s in spec]
    if all(is_int):
        return _validate_indices([int(s) for s in spec], n_features)
    if not all(isinstance(s, str) for s in spec):
        raise ValueError(
            "Categorical columns must be either all names (str) or all "
            "positional indices (int)."
        )
    return _names_to_indices(spec, columns)


def _validate_indices(indices: typing.List[int], n_features: int) -> typing.List[int]:
    invalid = [i for i in indices if not 0 <= i < n_features]
    if invalid:
        raise ValueError(
            f"Column indices {invalid} are out of range for {n_features} columns."
        )
    return sorted(indices)


def _names_to_indices(
    names: typing.List[str], columns: typing.Optional[typing.Sequence]
) -> typing.List[int]:
    if columns is None:
        raise ValueError(
            "Columns were selected by name, so X must be a pandas DataFrame."
        )
    position = {name: i for i, name in enumerate(columns)}
    missing = [name for name in names if name not in position]
    if missing:
        raise ValueError(f"Columns {missing} were not found in X.")
    return sorted(position[name] for name in names)


def _factorize(column: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    try:
        return pd.factorize(column, sort=True)
    except TypeError:  # unorderable mixed types
        return pd.factorize(column)


def encode_categorical(
    data: np.ndarray,
) -> typing.Tuple[np.ndarray, typing.List[np.ndarray]]:
    """Encode every column of an object array into integer codes.

    Returns the codes and, per column, the array of categories (``categories[j][c]``
    is the original value of code ``c`` in column ``j``).
    """
    codes = np.empty(data.shape, dtype=np.int64)
    categories = []
    for j in range(data.shape[1]):
        codes[:, j], uniques = _factorize(data[:, j])
        categories.append(np.asarray(uniques, dtype=object))
    return codes, categories


def apply_encoding(data: np.ndarray, categories: typing.List[np.ndarray]) -> np.ndarray:
    """Encode ``data`` with fitted categories; unseen values get a fresh code."""
    codes = np.empty(data.shape, dtype=np.int64)
    for j, cats in enumerate(categories):
        found = pd.Index(cats).get_indexer(data[:, j])
        codes[:, j] = np.where(found < 0, len(cats), found)
    return codes


def decode_categorical(
    codes: np.ndarray, categories: typing.List[np.ndarray]
) -> np.ndarray:
    """Inverse of :func:`encode_categorical` for a 2D array of codes."""
    out = np.empty(codes.shape, dtype=object)
    for j, cats in enumerate(categories):
        out[:, j] = cats[codes[:, j]]
    return out
