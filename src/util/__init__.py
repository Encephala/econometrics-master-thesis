from dataclasses import replace

from IPython.display import display
import pandas as pd
import semopy

from .data import Column, cleanup_dummy


def make_dummies(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([_series_to_dummies(df[column]) for column in df], axis=1)


def _series_to_dummies(series: pd.Series) -> pd.DataFrame:
    result = pd.get_dummies(series, prefix_sep=".", dtype="boolean")

    # Fix column names
    old_column: Column = series.name  # type: ignore  # noqa: PGH003
    result.columns = [replace(old_column, dummy_level=cleanup_dummy(dummy_level)) for dummy_level in result]  # type: ignore noqa: PGH003

    # If original was NA, the result will have False in each column.
    # Replace that False with NA
    where_original_na = series.isna()
    result.loc[where_original_na, :] = pd.NA

    return result


def print_results(model: semopy.Model):
    # https://stackoverflow.com/a/20937592/5410751
    results = model.inspect().sort_values(["op", "rval"])  # pyright: ignore noqa: PGH003

    with pd.option_context("display.float_format", "{:.4f}".format, "display.max_rows", None):
        display(results)
