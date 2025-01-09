#!/usr/bin/env python3

from pathlib import Path

import spss_converter
import pandas as pd

def load_df(path: Path) -> pd.DataFrame:
    full_path = Path("data") / path

    if path.suffix == ".csv":
        return pd.read_csv(full_path, sep = ";")

    if path.suffix == ".sav":
        return spss_converter.to_dataframe(full_path)[0]

    raise NotImplementedError

if __name__ == "__main__":
    test_df = load_df(Path("ai09e_EN_1.0p.csv"))

    print(test_df)
