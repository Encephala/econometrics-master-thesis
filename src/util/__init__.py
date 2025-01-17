import pandas as pd

from .data_load import load_df

def strip_column_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    """Takes all columns that represent a questionnaire question (with a wacky heuristic),
    and removes the questionnaire prefix."""
    columns = df.columns

    new_columns = []

    for col in columns:
        length = len(col)

        if col[length - 3:].isnumeric():
            new_columns.append(col[length - 3:])
        else:
            new_columns.append(col)

    df.columns = new_columns

    return df
