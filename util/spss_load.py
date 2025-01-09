#!/usr/bin/env python3

from pathlib import Path

import spss_converter
import pandas as pd

def load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".sav":
        return spss_converter.to_dataframe(path)[0]

    raise NotImplementedError

if __name__ == "__main__":
    test_df = load_df(Path("data/ai09e_EN_1.0p.csv"))

    print(test_df)
