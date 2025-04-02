import logging

import pandas as pd
import numpy as np

from util.data import Column

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


def select_wave(df: pd.DataFrame, year: int) -> pd.DataFrame:
    assert_column_type_correct(df)

    columns: list[Column] = df.columns  # pyright: ignore[reportAssignmentType]

    selected_columns = [column for column in columns if column.wave == year]

    if len(selected_columns) == 0:
        logger.warning(f"No columns selected for {year=}")

    return df[selected_columns]


# This assumes columns named as in standardise_wide_column_name above
def available_years(df: pd.DataFrame) -> set[int]:
    assert_column_type_correct(df)

    columns: list[Column] = df.columns  # pyright: ignore[reportAssignmentType]

    return {column.wave for column in columns if column.wave is not None}


def available_dummy_levels(df: pd.DataFrame, variable: str) -> set[str]:
    assert_column_type_correct(df)

    subset = select_variable(df, variable)

    columns: list[Column] = subset.columns  # pyright: ignore[reportAssignmentType]

    result = {column.dummy_level for column in columns if column.dummy_level is not None}

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


def map_mhi5_categories(series: pd.Series, *, is_positive: bool = False) -> pd.Series:
    "Takes a Categorical series of MHI-5 questionnaire responses and maps the textual responses to int values."
    # https://www.cbs.nl/nl-nl/achtergrond/2015/18/beperkingen-in-dagelijkse-handelingen-bij-ouderen/mhi-5
    # LISS response run from 1-6 for all questions
    # TODO: I'm not 100% this doesn't map NA to -1
    result = series.map(
        {
            "continuously": 0,
            "mostly": 1,
            "often": 2,
            "sometimes": 3,
            "seldom": 4,
            "never": 5,
        },
        na_action="ignore",
    ).astype("Int64")

    if is_positive:
        result = 5 - result

    return result


def calc_mhi5(health_panel: pd.DataFrame) -> pd.DataFrame:
    """Calculates the MHI-5 score for each individual-year available in the passed dataframe.

    Returns the MHI-5 scores in a new df (does not mutate the passed df)."""
    assert_column_type_correct(health_panel)

    # Appropriate question indices
    ANXIOUS = "ch11"
    CANT_CHEER_UP = "ch12"
    CALM_PEACEFUL = "ch13"
    DEPRESSED_GLOOMY = "ch14"
    HAPPY = "ch15"

    # The epic calculation
    result = pd.DataFrame()

    for year in available_years(health_panel):
        # These contain the responses of all individuals for this `year`
        anxious = map_mhi5_categories(health_panel[Column(ANXIOUS, year)])
        cheer_up = map_mhi5_categories(health_panel[Column(CANT_CHEER_UP, year)])
        calm = map_mhi5_categories(health_panel[Column(CALM_PEACEFUL, year)], is_positive=True)
        depressed = map_mhi5_categories(health_panel[Column(DEPRESSED_GLOOMY, year)])
        happy = map_mhi5_categories(health_panel[Column(HAPPY, year)], is_positive=True)

        # NOTE: There is no partial missingness, as in whenever one variable is missing,
        # all are missing. No need to f.i. impute NAs as the average of the other variables
        mhi = 4 * (anxious + cheer_up + calm + depressed + happy)

        result[Column("mhi", year)] = mhi

    return result
