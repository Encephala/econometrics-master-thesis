import warnings

import pandas as pd


def select_variable_wide(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    selected_columns = []

    for column in df.columns:
        if column == variable:
            selected_columns.append(column)
            continue

        if (index := column.rfind("_")) != -1 and column[:index] == variable:
            selected_columns.append(column)
            continue

    if len(selected_columns) == 0:
        warnings.warn(f"No columns selected for {variable=}", stacklevel=2)

    return df[selected_columns]


def standardise_wide_column_name(column_name: str) -> str:
    """Simple function to standardise column names, e.g. "cs12e005" -> "cs5_12".

    That is, of the form [questionnaire][question]_[year]."""
    # Skip non-question columns, e.g. cs12e_m
    if not column_name[-3:].isnumeric():
        return column_name

    prefix = column_name[:2]
    year = int(column_name[2:4])
    question = int(column_name[-3:])

    return f"{prefix}{question}_{year}"


# This assumes columns named as in standardise_wide_column_name above
def available_years(df: pd.DataFrame) -> set[int]:
    result = set()

    for column in df.columns:
        index_underscore = column.rfind("_")

        if column[index_underscore + 1 :].isnumeric():
            result.add(int(column[index_underscore + 1 :]))

    return result


def available_dummy_levels(df: pd.DataFrame, variable: str) -> set[str]:
    subset = select_variable_wide(df, variable)

    result = {column[column.find(".") + 1 :] for column in subset.columns if column.find(".") != -1}

    if len(result) == 0:
        warnings.warn(f"No dummy levels found for {variable}", stacklevel=2)

    return result


def cleanup_dummy(name: str) -> str:
    "Takes a dummy level and replaces characters to make semopy accept it."
    safe_character = "."

    return name.replace(" ", safe_character).replace("-", safe_character)


def cleanup_dummy_column(column: str) -> str:
    index_separator = column.find(".")

    if index_separator == -1:
        return column

    return f"{column[:index_separator]}.{cleanup_dummy(column[index_separator + 1 :])}"


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
