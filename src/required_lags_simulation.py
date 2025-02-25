#!/usr/bin/env python3
# %%
import pandas as pd
import semopy

from util import simulation, ModelDefinitionBuilder

# %%
N = 50000
T = 7

generator = simulation.DataGenerator([0.5, 0.3], [1, 0.9, 0.8, 0.7], [0, -0.1, -0.2])

# %%
x, y = generator.generate((N, T))

# %%
complete_data = pd.DataFrame()
for t in range(T):
    complete_data[f"y_{t}"] = y[:, t]
    complete_data[f"x_{t}"] = x[:, t]

model_definition = (
    ModelDefinitionBuilder()
    .with_x("x", lag_structure=[0, 1, 2, 3])
    .with_y("y", lag_structure=[1, 2, 3, 4])
    .build(complete_data.columns)
)

print(f"Model definition:\n{model_definition}")

model = semopy.Model(model_definition)

# %% model estimation
optimisation_result = model.fit(complete_data)
print(optimisation_result)

model.inspect().sort_values(["op", "Estimate", "lval"])  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
