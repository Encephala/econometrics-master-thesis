from pathlib import Path
import warnings

import pandas as pd


def load_df(path: Path) -> pd.DataFrame:
    """Given the file name, loads it from the data directory."""
    full_path = Path("../data") / path

    if path.suffix == ".sav":
        result = pd.read_spss(full_path)

    elif path.suffix == ".dta":
        result = pd.read_stata(full_path)

    else:
        raise NotImplementedError(f"Invalid suffix {path.suffix}")

    return result.set_index("nomem_encr")


# I sure hope LISS actually uses a consistent prefix for its studies
def assemble_wide_panel(prefix: str) -> pd.DataFrame:
    """Loads all files starting with the given prefix and merges them into a wide-form dataframe.
    This is done on the assumption that there are no duplicate files containing the same column names."""
    result = pd.DataFrame()

    for file in Path("../data").glob(prefix + "*"):
        if file.suffix == ".pkl":
            warnings.warn(f"Assembling panel for {prefix=} but pickle file already exists ({file})", stacklevel=2)
            continue

        new_df = load_df(file)

        # To avoid duplicating columns (mainly 'nohouse_encr')
        new_columns = new_df.columns.difference(result.columns)

        result = result.merge(
            new_df[new_columns],
            how="outer",
            left_index=True,
            right_index=True,
        )

    if len(result) == 0:
        warnings.warn(f"No data loaded for {prefix=}", stacklevel=2)

    return result


def assemble_background_panel() -> pd.DataFrame:
    result = pd.DataFrame()

    for file in Path("../data").glob("avars*"):
        if file.suffix == ".pkl":
            warnings.warn(f"Assembling panel for avars but pickle file already exists ({file})", stacklevel=2)
            continue

        new_df = load_df(file)

        year = file.stem[len("avars_20") :][:2]
        assert year.isnumeric(), f"{year} is not numeric, wrong indices"

        new_df.columns = [f"{column}_{int(year)}" for column in new_df.columns]

        result = result.merge(
            new_df,
            how="outer",
            left_index=True,
            right_index=True,
        )

    if len(result) == 0:
        warnings.warn("No data loaded for avars", stacklevel=2)

    return result


def load_wide_panel_cached(prefix: str, *, respect_cache: bool = True) -> pd.DataFrame:
    """`assemble_wide_panel`, but checks for a cached version on file first,
    and creates this cache if it doesn't exist.

    Note to self: no automatic cache invalidation happens :^)."""

    path = Path(f"../data/{prefix}_wide.pkl")

    if respect_cache and path.exists():
        return pd.read_pickle(path)  # noqa: S301

    assembled = assemble_background_panel() if prefix == "avars" else assemble_wide_panel(prefix)

    if len(assembled) != 0:
        assembled.to_pickle(path)
    else:
        warnings.warn("Not writing empty cache", stacklevel=2)

    return assembled
