# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
estimates = [10.54, 9.68, 9.42, 9.24, 9.19, 9.19, 9.01, 8.93]
stds = [0.36, 0.38, 0.41, 0.44, 0.48, 0.41, 0.49, 0.44]

plt.figure()
plt.title("RMSPE for varying maximum AR lag")
plt.ylabel("RMSPE (MHI5)")
plt.xlabel("Number of AR lags")

plt.errorbar(np.arange(len(estimates)) + 1, estimates, yerr=stds, capsize=4)

index_min = np.argmin(estimates)
plt.axhline(estimates[index_min] + stds[index_min], color="red", alpha=0.5)

plt.show()

# %% Comparing 3 to 4 max lags on a more complex model
# Which uh, is kinda meaningless. 1-sigma rule has very different meaning here.
# The difference is comparable to the difference with only mhi5_23 though, so corroborates the choice in that regard.

estimates_complex = [10.73, 10.53]
stds_complex = [0.37, 0.37]

plt.figure()
plt.title("RMSPE for varying maximum AR lag")
plt.ylabel("RMSPE (MHI5)")
plt.xlabel("Number of AR lags")

plt.errorbar(np.arange(len(estimates_complex)) + 1, estimates_complex, yerr=stds_complex, capsize=4)

index_min = np.argmin(estimates_complex)
plt.axhline(estimates_complex[index_min] + stds_complex[index_min], color="red", alpha=0.5)

plt.show()

# %% Comparing X lags (distributed lags)
# Minimum lag 0

estimates_x = [9.25, 9.29, 9.29, 9.32, 9.26, 9.24, 9.09, 9.06, 9.03, 8.84, 8.85]
stds_x = [0.42, 0.46, 0.45, 0.45, 0.35, 0.44, 0.59, 0.49, 0.40, 0.57, 0.43]

plt.figure()
plt.title("RMSPE for varying distributed lag")
plt.ylabel("RMSPE (MHI5)")
plt.xlabel("Number of X lags")

plt.errorbar(np.arange(len(estimates_x)) + 1, estimates_x, yerr=stds_x, capsize=4)

index_min = np.argmin(estimates_x)
plt.axhline(estimates_x[index_min] + stds_x[index_min], color="red", alpha=0.5)

plt.show()
