# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
estimates = [10.54, 9.68, 9.42, 9.24, 9.19, 9.19, 9.01, 8.93]
stds = [0.36, 0.38, 0.41, 0.44, 0.48, 0.41, 0.49, 0.44]

# %%
plt.figure()
plt.title("RMSPE for varying maximum AR lag")
plt.ylabel("RMSPE (MHI5)")
plt.xlabel("Number of AR lags")

plt.errorbar(np.arange(len(estimates)) + 1, estimates, yerr=stds, capsize=4)

index_min = np.argmin(estimates)
plt.axhline(estimates[index_min] + stds[index_min], color="red", alpha=0.5)

plt.show()
