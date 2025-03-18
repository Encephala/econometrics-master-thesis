# %%
import pandas as pd
import numpy as np

import semopy

from util.model import ModelDefinitionBuilder, VariableDefinition

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
DELTA0 = 50
SIGMA = np.sqrt(3)

rng = np.random.default_rng()
epsilon = rng.normal(0, SIGMA, size=(N, T))

x = rng.normal(5, 3, size=(N, T))
# Correlate x with previous epsilon
# np.roll moves the previous values of epsilon to align with current values of x.
# afterwards only apply the correlation for available data, i.e. don't apply 4th lag to t=0.
x[:, 1:] += GAMMA1 * np.roll(epsilon, 1, axis=1)[:, 1:]
x[:, 2:] += GAMMA2 * np.roll(epsilon, 2, axis=1)[:, 2:]
x[:, 4:] += GAMMA4 * np.roll(epsilon, 4, axis=1)[:, 4:]

some_control = rng.normal(size=(N, T))

y = np.empty([N, T])
# TODO: if y_0 is a constant, the covariance matrix becomes non-PD.
# I think that's because I'm not properly treating y_0 as exogenous?
# Like it shouldn't be a problem?
# Hmm when fixing the variance of y_0 to be any constant in the model definition,
# the fitting still complains about non-PD'ness.
# Curious
y[:, 0] = DELTA0 * some_control[:, 0] + epsilon[:, 0]
y[:, 1] = ALPHA1 * y[:, 0] + BETA1 * x[:, 0] + DELTA0 * some_control[:, 0] + epsilon[:, 1]
y[:, 2] = ALPHA1 * y[:, 1] + ALPHA2 * y[:, 0] + BETA1 * x[:, 1] + DELTA0 * some_control[:, 0] + epsilon[:, 2]

for t in range(3, T):
    y[:, t] = (
        ALPHA1 * y[:, t - 1]
        + ALPHA2 * y[:, t - 2]
        + ALPHA3 * y[:, t - 3]
        + BETA1 * x[:, t - 1]
        + DELTA0 * some_control[:, t]
        + epsilon[:, t]
    )

# %% model definition
complete_data = pd.DataFrame()
for t in range(T):
    complete_data[f"y_{t}"] = y[:, t]
    complete_data[f"x_{t}"] = x[:, t]
    complete_data[f"some_control_{t}"] = some_control[:, t]

model_definition = (
    ModelDefinitionBuilder()
    .with_x(VariableDefinition("x"))
    .with_y(VariableDefinition("y"), lag_structure=[1, 2, 3])
    .with_w([VariableDefinition("some_control")])
    .build(complete_data.columns)
)

print(f"Model definition:\n{model_definition}")

model = semopy.Model(model_definition)

# %% model estimation
optimisation_result = model.fit(complete_data)
print(optimisation_result)

model.inspect().sort_values(["op", "Estimate", "lval"])  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
