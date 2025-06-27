# %%
import matplotlib.pyplot as plt

from lib.data import available_waves, make_all_data, select_variable

# ruff: noqa: F403, F405
from lib.data.variables import *

data = make_all_data(cache=True)

# %% MHI5
mhi5 = select_variable(data, MHI5)
mhi5_means = mhi5.mean()

mhi5_years = [f"20{year}" for year in list(available_waves(mhi5))]

plt.figure()
plt.plot(mhi5_years, mhi5_means, "-o")
plt.title("Mean MHI5 score over time")
plt.ylim(0, 100)
plt.ylabel("MHI5")
plt.xlabel("Year")
plt.show()

# %% Examine std
mhi5_stds = mhi5.std()

plt.figure()
plt.plot(mhi5_years, mhi5_stds, "-o")
plt.title("Std of MHI5 score over time")
plt.ylim(0, 30)
plt.ylabel(r"$\sigma_{MHI5}$")
plt.xlabel("Year")
plt.show()

# %% Examine sports
sports = select_variable(data, SPORTS)
sports_means = sports.mean()

sports_years = [f"20{year}" for year in list(available_waves(sports))]

plt.figure()
plt.plot(sports_years, sports_means, "-o")
plt.title("Sports engagement rate over time")
plt.ylim(0, 1)
plt.ylabel(r"$p_{sports}$")
plt.xlabel("Year")
plt.show()

# %% Sports std
sports_stds = sports.std()

plt.figure()
plt.plot(sports_years, sports_stds, "-o")
plt.title("Std of MHI5 score over time")
plt.ylim(0, 1)
plt.ylabel(r"$\sigma_{p_{sports}}$")
plt.xlabel("Year")
plt.show()

# %% Combined plots
mhi5_no_nan = [mhi5[col].dropna() for col in mhi5]

plt.figure(figsize=(4, 3))
plt.boxplot(mhi5_no_nan, positions=list(available_waves(mhi5)), whis=(10, 90), showfliers=False)
plt.title("MHI5 score distribution over time")
plt.ylim(-4, 104)
plt.ylabel("MHI5")
plt.xlabel("Year")
plt.xticks(list(range(8, 24, 2)), labels=list(range(8, 24, 2)))  # pyright: ignore  # noqa: PGH003
plt.savefig("../report/thesis/figures/data/boxplot_mhi5.svg")

plt.figure(figsize=(4, 3))
plt.plot(list(available_waves(sports)), sports_means, ".-", color="black")
plt.title("Sports engagement over time")
plt.ylim(0, 1)
plt.ylabel("Rate of engagement in sports")
plt.xlabel("Year")
plt.savefig("../report/thesis/figures/data/errorbar_sports.svg")

plt.show()
