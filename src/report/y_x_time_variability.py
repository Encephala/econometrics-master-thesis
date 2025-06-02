# %%
import matplotlib.pyplot as plt

from lib.data import available_waves, make_all_data, select_variable

# ruff: noqa: F403, F405
from lib.data.variables import *

data = make_all_data(cache=True)

# %% MHI5
mhi5 = select_variable(data, MHI5)
mhi5_means = mhi5.mean()

plt.figure()
plt.plot(list(available_waves(mhi5)), mhi5_means, "-o")
plt.title("Mean MHI5-score over time")
plt.ylim(0, 100)
plt.ylabel("MHI5")
plt.xlabel("Year")
plt.show()

# %% Examine std
mhi5_stds = mhi5.std()

plt.figure()
plt.plot(list(available_waves(mhi5)), mhi5_stds, "-o")
plt.title("Std of MHI5-score over time")
plt.ylim(0, 30)
plt.ylabel(r"$\sigma_{MHI5}$")
plt.xlabel("Year")
plt.show()

# %% Examine sports
sports = select_variable(data, SPORTS)
sports_means = sports.mean()

plt.figure()
plt.plot(list(available_waves(sports)), sports_means, "-o")
plt.title("Sports engagement rate over time")
plt.ylim(0, 1)
plt.ylabel(r"$p_{sports}$")
plt.xlabel("Year")
plt.show()

# %% Sports std
sports_stds = sports.std()

plt.figure()
plt.plot(list(available_waves(sports)), sports_stds, "-o")
plt.title("Std of MHI5-score over time")
plt.ylim(0, 1)
plt.ylabel(r"$\sigma_{p_{sports}}$")
plt.xlabel("Year")
plt.show()

# %% Combined plots
mhi5_no_nan = [mhi5[col].dropna() for col in mhi5]

plt.figure(figsize=(4, 3))
plt.boxplot(mhi5_no_nan, positions=list(available_waves(mhi5)), showfliers=False)
plt.title("Mean MHI5-score over time")
plt.ylim(-4, 104)
plt.ylabel("MHI5")
plt.xlabel("Year")
plt.xticks(list(range(8, 24, 2)), labels=list(range(8, 24, 2)))  # pyright: ignore  # noqa: PGH003
plt.savefig("../report/thesis/figures/data/boxplot_mhi5.svg")

plt.figure(figsize=(4, 3))
plt.errorbar(list(available_waves(sports)), sports_means, yerr=sports_stds, capsize=2, color="black")
plt.title("Sports engagement over time")
plt.ylabel("Rate of engagement in sports")
plt.xlabel("Year")
plt.savefig("../report/thesis/figures/data/errorbar_sports.svg")

plt.show()
