from pathlib import Path

import pandas as pd


def select_question_wide(df: pd.DataFrame, question_id: str) -> pd.DataFrame:
    selected_columns = list(
        filter(lambda name: name[: name.find("_")] == question_id, df.columns)
    )

    if len(selected_columns) == 0:
        raise UserWarning("No columns selected")  # noqa: TRY003

    return df[selected_columns]


def standardise_wide_column(column_name: str) -> str:
    """Simple function to standardise column names, e.g. "cs12e001" -> "cs1_12"."""
    if not column_name[-3:].isnumeric():
        return column_name

    prefix = column_name[:2]
    year = int(column_name[2:4])
    question = int(column_name[-3:])

    return f"{prefix}{question}_{year}"

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

def load_df(path: Path) -> pd.DataFrame:
    """Given the file name, loads it from the data directory."""
    full_path = Path("../data") / path

    if path.suffix == ".sav":
        result = pd.read_spss(full_path)

    elif path.suffix == ".dta":
        result = pd.read_stata(full_path)

    else:
        raise NotImplementedError

    return result.set_index("nomem_encr")

# I sure hope LISS actually uses a consistent prefix for its studies
def assemble_wide_panel(prefix: str) -> pd.DataFrame:
    """Loads all files starting with the given prefix and merges them into a wide-form dataframe.
    This is done on the assumption that there are no duplicate files containing the same column names."""
    result = pd.DataFrame()

    for file in Path("../data").glob(prefix + "*"):
        new_df = load_df(file)

        # To avoid duplicating columns (mainly 'nohouse_encr')
        new_columns = new_df.columns.difference(result.columns)

        result = result.merge(
            new_df[new_columns],
            how = "outer",
            left_index = True,
            right_index = True,
        )

    return result

def load_wide_panel_cached(prefix: str) -> pd.DataFrame:
    """`assemble_wide_panel`, but checks for a cached version on file first,
    and creates this cache if it doesn't exist."""

    path = f"../data/{prefix}_wide.pkl"

    if Path(path).exists():
        return pd.read_pickle(path) # noqa: S301

    assembled = assemble_wide_panel(prefix)

    assembled.to_pickle(path)

    return assembled
