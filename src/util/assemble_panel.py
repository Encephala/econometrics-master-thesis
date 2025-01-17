#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

from data_load import load_df

# I sure hope LISS actually uses a consistent prefix for its studies
def assemble_wide_panel(prefix: str) -> pd.DataFrame:
    """Loads all files starting with the given prefix and merges them into a wide-form dataframe."""
    result = pd.DataFrame()

    for file in Path("../data").glob(prefix + "*"):
        result = result.merge(
            load_df(file),
            how = "outer",
            left_index = True,
            right_index = True,
        )

    return result

if __name__ == "__main__":
    df = assemble_wide_panel("cp")

    print(df)
    print(df.columns)
