# %%
import pandas as pd

from lib.data import make_all_data, select_variable

# ruff: noqa: F403, F405
from lib.data.variables import *

# %%

mhi5 = select_variable(make_all_data(cache=True), MHI5)

# %%
missing_percents = []

for _, row in mhi5.iterrows():
    first_nonna_idx = row.first_valid_index()
    last_nonna_idx = row.last_valid_index()

    if first_nonna_idx is None:
        continue

    in_between = row.loc[first_nonna_idx:last_nonna_idx]

    # Strip off edges since they are by definition not NA
    in_between = in_between.iloc[1:-1]

    if in_between.empty:
        continue

    missing_percents.append(in_between.isna().sum() / len(in_between))

(pd.Series(missing_percents) * 100).describe()
