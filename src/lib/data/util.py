import logging

import pandas as pd
import numpy as np

from lib.data import Column

logger = logging.getLogger(__name__)


def assert_column_type_correct(df: pd.DataFrame):
    for column in df.columns:
        assert isinstance(column, Column), f"{column=} is not of type `util.data.Column`"


def select_variable(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    assert_column_type_correct(df)

    columns: list[Column] = df.columns  # pyright: ignore[reportAssignmentType]

    selected_columns = [column for column in columns if column.name == variable]

    if len(selected_columns) == 0:
        logger.warning(f"No columns selected for {variable=}")

    return df[selected_columns]


def select_wave(df: pd.DataFrame, wave: int) -> pd.DataFrame:
    assert_column_type_correct(df)

    columns: list[Column] = df.columns  # pyright: ignore[reportAssignmentType]

    selected_columns = [column for column in columns if column.wave == wave]

    if len(selected_columns) == 0:
        logger.warning(f"No columns selected for {wave=}")

    return df[selected_columns]


def available_waves(df: pd.DataFrame) -> set[int]:
    assert_column_type_correct(df)

    columns: list[Column] = df.columns  # pyright: ignore[reportAssignmentType]

    return {column.wave for column in columns if column.wave is not None}


def available_dummy_levels(df: pd.DataFrame, variable: str) -> list[str]:
    assert_column_type_correct(df)

    subset = select_variable(df, variable)

    columns: list[Column] = subset.columns  # pyright: ignore[reportAssignmentType]

    result = [column.dummy_level for column in columns if column.dummy_level is not None]

    # Make unique, dicts maintain order as well nowadays
    result = list(dict.fromkeys(result))

    if len(result) == 0:
        logger.warning(f"No dummy levels found for {variable}")

    return result


def cleanup_dummy(name: str) -> str:
    "Takes a dummy level and replaces characters to make semopy accept it."
    safe_character = "."

    return name.replace(" ", safe_character).replace("-", safe_character)


def map_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(column) for column in df.columns]

    return df


def find_non_PD_suspicious_columns(df: pd.DataFrame) -> set[Column]:
    # https://stats.stackexchange.com/questions/153632/how-to-find-factor-that-is-making-matrix-singular

    assert_column_type_correct(df)

    result = set()

    columns: list[Column] = df.columns  # pyright: ignore[reportAssignmentType]

    Sigma = df.cov()

    assert np.allclose(Sigma, Sigma.T)

    eigvals, eigvecs = np.linalg.eigh(Sigma)

    for eigval, eigvec in zip(eigvals, eigvecs.T, strict=True):
        if not np.isclose(eigval, 0):
            # All good
            continue

        eigvec = pd.Series(eigvec, index=columns)  # noqa: PLW2901

        suspicious_columns = eigvec[~np.isclose(eigvec, 0)].index

        result.update(suspicious_columns)

    return result
