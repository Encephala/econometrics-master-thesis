# %%
import matplotlib.pyplot as plt

from lib.data import available_waves, make_all_data, select_variable

# ruff: noqa: F403, F405
from lib.data.variables import *

data = make_all_data(cache=True)

# %% MHI5
mhi5 = select_variable(data, MHI5)
means = mhi5.mean()

plt.figure()
plt.plot(list(available_waves(mhi5)), means, "-o")
plt.title("Mean MHI5-score over time")
plt.ylim(0, 100)
plt.ylabel("MHI5")
plt.xlabel("Year")
plt.show()

# %% Examine std
stds = mhi5.std()

plt.figure()
plt.plot(list(available_waves(mhi5)), stds, "-o")
plt.title("Std of MHI5-score over time")
plt.ylim(0, 30)
plt.ylabel(r"$\sigma_{MHI5}$")
plt.xlabel("Year")
plt.show()

# %% Examine sports
sports = select_variable(data, SPORTS)
means = sports.mean()

plt.figure()
plt.plot(list(available_waves(sports)), means, "-o")
plt.title("Sports engagement rate over time")
plt.ylim(0, 1)
plt.ylabel(r"$p_{sports}$")
plt.xlabel("Year")
plt.show()

# %% Sports std
stds = sports.std()

plt.figure()
plt.plot(list(available_waves(sports)), stds, "-o")
plt.title("Std of MHI5-score over time")
plt.ylim(0, 1)
plt.ylabel(r"$\sigma_{p_{sports}}$")
plt.xlabel("Year")
plt.show()
