# %%
import pandas as pd
import numpy as np

import semopy

from util import ModelDefinitionBuilder

# %% data generation
N = 500
T = 5

RHO = 0
BETA = 0.69

epsilon = np.random.normal(0, 1, size=(N, T))  # noqa: NPY002

# TODO: Correlate x with the previous epsilon (reverse causality)
x = np.array(list(range(N)) * T).reshape((N, T)) + np.random.normal(0, 2, size=(N, T))  # noqa: NPY002

y = np.empty((N, T))
y[:, 0] = 0

for t in range(1, T):
    y[:, t] = RHO * y[:, t - 1] + BETA * x[:, t - 1] + epsilon[:, t]

# %% model definition
complete_data = pd.DataFrame()
for t in range(T):
    complete_data[f"y_{t}"] = y[:, t]
    complete_data[f"x_{t}"] = x[:, t]

lag_structure = [1]
model_definition = (
    ModelDefinitionBuilder().with_x("x").with_y("y").with_lag_structure(lag_structure).build(complete_data.columns)
)

print(f"Model definition:\n{model_definition}")

model = semopy.Model(model_definition)

# %% model estimation
optimisation_result = model.fit(complete_data)
print(optimisation_result)

model.inspect()
