# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
estimates = [10.47, 9.5, 9.40, 9.23, 9.18, 9.18, 9.00, 8.91]
stds = [0.35, 0.37, 0.41, 0.44, 0.48, 0.40, 0.49, 0.44]

# %%
plt.figure()
plt.title("RMSPE for varying maximum AR lag")
plt.ylabel("RMSPE")
plt.xlabel("Number of AR lags")

plt.errorbar(np.arange(len(estimates)) + 1, estimates, yerr=stds, capsize=4)

index_min = np.argmin(estimates)
plt.axhline(estimates[index_min] + stds[index_min], color="red", alpha=0.5)

plt.show()
