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
figure, axes = plt.subplots(1, 2, figsize=(9, 4))

mhi5_no_nan = [mhi5[col].dropna() for col in mhi5]

axes[0].boxplot(mhi5_no_nan, positions=list(available_waves(mhi5)), showfliers=False)
axes[0].set_title("Mean MHI5-score over time")
axes[0].set_ylim(-4, 104)
axes[0].set_ylabel("MHI5")
axes[0].set_xlabel("Year")

axes[1].errorbar(list(available_waves(sports)), sports_means, yerr=sports_stds, capsize=2, color="black")
axes[1].set_title("Sports engagement over time")
axes[1].set_ylabel("Rate of engagement in sports")
axes[1].set_xlabel("Year")

plt.tight_layout()
plt.savefig("../report/thesis/figures/data/errorbar_mhi5_sports.svg")
plt.show()
