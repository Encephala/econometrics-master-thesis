import warnings

import pandas as pd

from util.data import Column


def select_variable_wide(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    columns: list[Column] = df.columns  # pyright: ignore[reportAssignmentType]

    selected_columns = [column for column in columns if column.name == variable]

    if len(selected_columns) == 0:
        warnings.warn(f"No columns selected for {variable=}", stacklevel=2)

    return df[selected_columns]


def select_wave_wide(df: pd.DataFrame, year: int) -> pd.DataFrame:
    columns: list[Column] = df.columns  # pyright: ignore[reportAssignmentType]

    selected_columns = [column for column in columns if column.wave == year]

    if len(selected_columns) == 0:
        warnings.warn(f"No columns selected for {year=}", stacklevel=2)

    return df[selected_columns]


# This assumes columns named as in standardise_wide_column_name above
def available_years(df: pd.DataFrame) -> set[int]:
    columns: list[Column] = df.columns  # pyright: ignore[reportAssignmentType]

    return {column.wave for column in columns if column.wave is not None}


def available_dummy_levels(df: pd.DataFrame, variable: str) -> set[str]:
    subset = select_variable_wide(df, variable)

    columns: list[Column] = subset.columns  # pyright: ignore[reportAssignmentType]

    result = {column.dummy_level for column in columns if column.dummy_level is not None}

    if len(result) == 0:
        warnings.warn(f"No dummy levels found for {variable}", stacklevel=2)

    return result


def cleanup_dummy(name: str) -> str:
    "Takes a dummy level and replaces characters to make semopy accept it."
    safe_character = "."

    return name.replace(" ", safe_character).replace("-", safe_character)


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


def calc_mhi5(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the MHI-5 score for each individual-year available in the passed dataframe.

    Returns the MHI-5 scores in a new df (does not mutate the passed df)."""

    # Appropriate question indices
    ANXIOUS = "11"
    CHEER_UP = "12"
    CALM_PEACEFUL = "13"
    DEPRESSED_GLOOMY = "14"
    HAPPY = "15"

    # The epic calculation
    result = pd.DataFrame()

    for year in available_years(df):
        # TODO: Check for partial missingness
        # If present, don't have mhi=NA but scale partial calculation(?)

        # These contain the responses of all individuals for this `year`
        anxious = map_mhi5_categories(df[f"ch{ANXIOUS}_{year}"])
        cheer_up = map_mhi5_categories(df[f"ch{CHEER_UP}_{year}"])
        calm = map_mhi5_categories(df[f"ch{CALM_PEACEFUL}_{year}"], is_positive=True)
        depressed = map_mhi5_categories(df[f"ch{DEPRESSED_GLOOMY}_{year}"])
        happy = map_mhi5_categories(df[f"ch{HAPPY}_{year}"], is_positive=True)

        mhi = 4 * (anxious + cheer_up + calm + depressed + happy)

        result[f"mhi_{year}"] = mhi

    return result
