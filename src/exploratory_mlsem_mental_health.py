#!/usr/bin/env python3

# %% imports
import semopy
import pandas as pd

from util.model import ModelDefinitionBuilder, VariableDefinition
from util.data import (
    load_wide_panel_cached,
    standardise_wide_column_name,
    select_variable,
)


# %% consts
HAPPINESS = "ch15"
FITNESS = "cs121"

# %% data loading
leisure_panel = load_wide_panel_cached("cs").rename(columns=standardise_wide_column_name)
health_panel = load_wide_panel_cached("ch").rename(columns=standardise_wide_column_name)


# %% selecting columns

happiness = select_variable(health_panel, HAPPINESS)
happiness = happiness.apply(
    lambda column: pd.Categorical(
        column,
        categories=["never", "seldom", "sometimes", "often", "mostly", "continuously"],
        ordered=True,
    ),  # pyright: ignore[reportCallIssue, reportArgumentType]
)

fitness = select_variable(leisure_panel, FITNESS)
fitness = fitness.apply(
    lambda column: pd.Categorical(column, categories=["no", "yes"], ordered=True)  # pyright: ignore[reportCallIssue, reportArgumentType]
)

# %% build simple semopy model
complete_data = pd.concat([happiness, fitness], join="outer", axis="columns").apply(
    lambda column: pd.Series(column).cat.codes.map(lambda value: float("nan") if value == -1 else value)
)
model_definition = (
    ModelDefinitionBuilder()
    .with_y(VariableDefinition(HAPPINESS), lag_structure=[])
    .with_x(VariableDefinition(FITNESS), lag_structure=[0, 1, 2, 3, 4])
    .build(complete_data.columns)
)

print(model_definition)

model = semopy.Model(model_definition)

# %% fit the model
optimisation_result = model.fit(complete_data)

print(optimisation_result)

model.inspect().sort_values(["op", "Estimate", "lval"])  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
