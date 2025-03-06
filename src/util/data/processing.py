import warnings

import pandas as pd


def select_question_wide(df: pd.DataFrame, question_id: str) -> pd.DataFrame:
    selected_columns = list(filter(lambda name: name[: name.find("_")] == question_id, df.columns))

    if len(selected_columns) == 0:
        warnings.warn("No columns selected", stacklevel=2)

    return df[selected_columns]


def standardise_wide_column_name(column_name: str) -> str:
    """Simple function to standardise column names, e.g. "cs12e005" -> "cs5_12"."""
    # Skip non-question columns, e.g. cs12e_m
    if not column_name[-3:].isnumeric():
        return column_name

    prefix = column_name[:2]
    year = int(column_name[2:4])
    question = int(column_name[-3:])

    return f"{prefix}{question}_{year}"


def fix_column_categories(column: pd.Series) -> pd.Series:
    """Actually have to do this because the data sometimes (but not consistently, smile) has prefixed spaces,
    and sometimes has capitalisation.
    I think 2013 and before there was a prefix space, after there wasn't."""

    old_categories: "pd.Index[str]" = column.cat.categories  # To help the LSP
    new_categories = pd.Index([category.lower().strip() for category in old_categories])

    return column.cat.rename_categories(new_categories)


def map_mhi5_categories(series: pd.Series, *, is_positive: bool = False) -> pd.Series:
    series = fix_column_categories(series)

    # https://www.cbs.nl/nl-nl/achtergrond/2015/18/beperkingen-in-dagelijkse-handelingen-bij-ouderen/mhi-5
    # LISS response run from 1-6 for all questions
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

    # result will have Int64 type with NaN preserved
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

    # Find available years
    available_years = {int(column[-2:]) for column in df.columns if column[-2:].isnumeric()}

    # The epic calculation
    result = pd.DataFrame()

    for year in available_years:
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
