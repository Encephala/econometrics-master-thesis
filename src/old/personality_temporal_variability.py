#!/usr/bin/env python3

#%% Imports
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from util import load_df, strip_column_prefixes

#%% Load data
data_2018 = load_df(Path("cp18j_EN_1.0p.sav"))
data_2024 = load_df(Path("cp24p_EN_1.0p.sav"))

# Standardise column names
data_2018 = strip_column_prefixes(data_2018)
data_2024 = strip_column_prefixes(data_2024)

# Drop non-numeric
data_2018 = data_2018.loc[:, data_2018.dtypes == "float64"]\
    .drop(columns = ["184", "185", "186", "187", "188", "193"])
data_2024 = data_2024.loc[:, data_2024.dtypes == "float64"]\
    .drop(columns = ["184", "185", "186", "187", "188", "193"])


#%% (very) Rough idea of what the typical change is in responses
results: list[pd.Series] = []

for user in data_2018.index:
    if user not in data_2024.index:
        continue

    left_info: pd.Series = data_2018.loc[user]
    right_info: pd.Series = data_2024.loc[user]

    results.append((left_info - right_info).dropna())

stds = [result.std() for result in results if result.std() is not np.nan and result.std() < 50]
print(stds)
print(np.mean(stds))

plt.figure()
plt.hist(stds)
plt.show()

#%% Difference over six years in question 010, "On the whole, how happy would you say you are?"

diffs: list[float] = []

for user in data_2018.index:
    if user not in data_2024.index:
        continue

    diff: float = data_2018.loc[user, "010"] - data_2024.loc[user, "010"] # type: ignore
    # Data is from 1 to 10, 999 is outlier (not filled in or something)
    if diff < 10:
        diffs.append(diff)

plt.figure()
plt.hist(diffs)
plt.show()
