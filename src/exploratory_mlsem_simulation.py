# %%
import pandas as pd
import numpy as np

# %%
N = 500
T = 5

x = np.array(list(range(N)) * T).reshape((N, T)) + np.random.normal(0, 2, size=(N, T))  # noqa: NPY002


# %%
RHO = 0.42
BETA = 0.69

y = np.empty((N, T))
y[:, 0] = 0

for t in range(1, T):
    y[:, t] = RHO * y[:, t - 1] + BETA * x[:, t - 1] + np.random.normal(0, 1, size=(N))  # noqa: NPY002

# %%
