# %%
import pandas as pd
import numpy as np

import semopy

from util import ModelDefinitionBuilder

# %% data generation
N = 500
T = 7

RHO = 0.42
BETA0 = 0.69
BETA1 = 0.75
BETA2 = 0.20
BETA4 = 1

rng = np.random.default_rng()
epsilon = rng.normal(0, np.sqrt(3), size=(N, T))

x = rng.normal(5, 3, size=(N, T))
x[:, 1:] += BETA1 * np.roll(epsilon, 1, axis=1)[:, 1:]
x[:, 2:] += BETA2 * np.roll(epsilon, 2, axis=1)[:, 2:]
x[:, 4:] += BETA4 * np.roll(epsilon, 4, axis=1)[:, 4:]

y = np.empty([N, T])
# TODO: if y_0 is a constant, the covariance matrix becomes non-PD.
# I think that's because I'm not properly treating y_0 as exogenous?
# Like it shouldn't be a problem?
# Hmm when fixing the variance of y_0 to be any constant, the model still complains about non-PD'ness.
# Curious
y[:, 0] = rng.normal(0, 0.05, size=N)

for t in range(1, T):
    y[:, t] = RHO * y[:, t - 1] + BETA1 * x[:, t - 1] + epsilon[:, t]

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
