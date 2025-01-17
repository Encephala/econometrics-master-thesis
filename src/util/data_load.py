#!/usr/bin/env python3

from pathlib import Path

import pandas as pd

def load_df(path: Path) -> pd.DataFrame:
    """Given the file name, loads it from the data directory."""
    full_path = Path("../data") / path

    if path.suffix == ".csv":
        result = pd.read_csv(full_path, sep = ";")

    elif path.suffix == ".sav":
        result = pd.read_spss(full_path)

    elif path.suffix == ".dta":
        result = pd.read_stata(full_path)

    else:
        raise NotImplementedError

    return result.set_index("nomem_encr")

if __name__ == "__main__":
    for test_file in ["ai09e_EN_1.0p.sav", "ch23p_EN_1.0p.csv", "cp24p_EN_1.0p.dta"]:
        test_df = load_df(Path(test_file))
        print(test_df)
        print(test_df.dtypes)
