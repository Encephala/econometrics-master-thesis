#!/usr/bin/env python3

# %% imports
import semopy
import pandas as pd

from lib import print_results
from lib.model import PanelModelDefinitionBuilder, VariableDefinition
from lib.data import (
    load_wide_panel_cached,
    select_variable,
    map_columns_to_str,
)


# %% consts
HAPPINESS = "ch15"
FITNESS = "cs121"

# %% data loading
leisure_panel = load_wide_panel_cached("cs")
health_panel = load_wide_panel_cached("ch")


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
    PanelModelDefinitionBuilder()
    .with_y(VariableDefinition(HAPPINESS), lag_structure=[])
    .with_x(VariableDefinition(FITNESS), lag_structure=[0, 1, 2, 3, 4])
    .build(complete_data)
)

print(model_definition)

model = semopy.Model(model_definition)

# %% fit the model
model.fit(map_columns_to_str(complete_data))

print_results(model)
