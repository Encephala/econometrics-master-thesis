# %%
import pandas as pd
import numpy as np

import semopy

from util import ModelDefinitionBuilder

# %% data generation
N = 500
T = 7

ALPHA1 = 0.42
ALPHA2 = 0.1234
ALPHA3 = -0.4321
BETA1 = 1.69
GAMMA1 = 0.75
GAMMA2 = 0.20
GAMMA4 = 1

rng = np.random.default_rng()
epsilon = rng.normal(0, np.sqrt(3), size=(N, T))

x = rng.normal(5, 3, size=(N, T))
# Correlate x with previous epsilon
x[:, 1:] += GAMMA1 * np.roll(epsilon, 1, axis=1)[:, 1:]
x[:, 2:] += GAMMA2 * np.roll(epsilon, 2, axis=1)[:, 2:]
x[:, 4:] += GAMMA4 * np.roll(epsilon, 4, axis=1)[:, 4:]

y = np.empty([N, T])
# TODO: if y_0 is a constant, the covariance matrix becomes non-PD.
# I think that's because I'm not properly treating y_0 as exogenous?
# Like it shouldn't be a problem?
# Hmm when fixing the variance of y_0 to be any constant in the model definition,
# the fitting still complains about non-PD'ness.
# Curious
y[:, 0] = rng.normal(0, 0.05, size=N)
y[:, 1] = ALPHA1 * y[:, 0] + BETA1 * x[:, 0] + epsilon[:, 1]
y[:, 2] = ALPHA1 * y[:, 1] + ALPHA2 * y[:, 0] + BETA1 * x[:, 1] + epsilon[:, 2]
y[:, 3] = ALPHA1 * y[:, 2] + ALPHA2 * y[:, 1] + ALPHA3 * y[:, 0] + BETA1 * x[:, 2] + epsilon[:, 3]

for t in range(4, T):
    y[:, t] = ALPHA1 * y[:, t - 1] + ALPHA2 * y[:, t - 2] + ALPHA3 * y[:, t - 3] + BETA1 * x[:, t - 1] + epsilon[:, t]

# %% model definition
complete_data = pd.DataFrame()
for t in range(T):
    complete_data[f"y_{t}"] = y[:, t]
    complete_data[f"x_{t}"] = x[:, t]

model_definition = (
    ModelDefinitionBuilder().with_x("x").with_y("y", lag_structure=[1, 2, 3]).build(complete_data.columns)
)

print(f"Model definition:\n{model_definition}")

model = semopy.Model(model_definition)

# %% model estimation
optimisation_result = model.fit(complete_data)
print(optimisation_result)

model.inspect().sort_values(["op", "Estimate", "lval"])  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
