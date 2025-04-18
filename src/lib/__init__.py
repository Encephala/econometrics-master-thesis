from pathlib import Path

from IPython.display import display
import pandas as pd
import numpy as np
import semopy
import pyperclip

from lib.data import map_columns_to_str


def print_results(model: semopy.Model):
    results = model.inspect().sort_values(["op", "rval"])  # pyright: ignore # noqa: PGH003

    # https://stackoverflow.com/a/20937592/5410751
    with pd.option_context("display.float_format", "{:.4f}".format, "display.max_rows", None):
        display(results)


def save_for_R(model: str, data: pd.DataFrame, path: Path):
    data_flattened = map_columns_to_str(data.astype(np.float64))

    # Manually replace . by _ to avoid a huge warning from stat
    data_flattened.columns = [column.replace(".", "_") for column in data_flattened.columns]
    data_flattened.to_stata(path)

    # Because the stata file format doesn't allow "." in variable names
    print("Model definition in stata/lavaan form:")
    print(model.replace(".", "_"))

    pyperclip.copy(model.replace(".", "_"))
    print("Copied model to clipboard.")
