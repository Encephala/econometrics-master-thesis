# %%
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps

from lib.data import make_all_data, select_variable

# ruff: noqa: F403, F405
from lib.data.variables import *

# %%
all_data = make_all_data(cache=True)


# %%
def check_normality(df: pd.DataFrame):
    for col in df:
        column = df[col].dropna().astype("float64")

        print(f"{col=}")
        print(f"Skewness: {sps.skew(column)}")
        print(f"Kurtosis: {sps.kurtosis(column)}")

        print(sps.jarque_bera(column.dropna()))

        plt.figure()
        plt.title(col)
        column.hist()
        plt.show()


# %% x
mhi5 = select_variable(all_data, MHI5)

check_normality(mhi5)

# %%
sports = select_variable(all_data, SPORTS)

check_normality(sports)
