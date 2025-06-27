# %%
from lib.data import make_all_data, available_waves, Column, select_variable

# ruff: noqa: F403, F405
from lib.data.variables import *

data = make_all_data(cache=True)

# %% Remove variables as in ../path_analysis.py
# Drop rows for which the dependent variable is always NA, as these will never be included in a regression.
y = select_variable(data, MHI5)
y_missing = y.isna().sum(axis=1) == y.shape[1]
y_missing = y_missing[y_missing].index
data = data.drop(y_missing)

# Drop rows for which sports is always NA, as these provide no information towards our variable of interest
x = select_variable(data, SPORTS)
x_missing = x.isna().sum(axis=1) == x.shape[1]
x_missing = x_missing[x_missing].index
data = data.drop(x_missing)

# %%
for wave in available_waves(data):
    print(f"{wave=}")

    try:
        mhi5 = data[Column(MHI5, wave)]
    except:  # noqa: E722
        print("Not found mhi5")
        continue

    try:
        sports = data[Column(SPORTS, wave)]
    except:  # noqa: E722
        print("Not found sports")
        continue

    coverage = (~mhi5.isna()) & (~sports.isna())

    print(f"{coverage.mean():.2%}")
