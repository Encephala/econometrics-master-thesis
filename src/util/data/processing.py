import warnings

import pandas as pd

from exploratory_mlsem_mental_health import HAPPINESS


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
