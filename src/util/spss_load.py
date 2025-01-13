#!/usr/bin/env python3

from pathlib import Path

import spss_converter
import pandas as pd

def load_df(path: Path) -> pd.DataFrame:
    """Given the file name, loads it from the data directory."""
    full_path = Path("../data") / path

    if path.suffix == ".csv":
        result = pd.read_csv(full_path, sep = ";")

    elif path.suffix == ".sav":
        result = spss_converter.to_dataframe(full_path)[0]

    else:
        raise NotImplementedError

    return result.set_index("nomem_encr")

if __name__ == "__main__":
    test_df = load_df(Path("ai09e_EN_1.0p.csv"))

    print(test_df)
