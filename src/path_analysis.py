#!/usr/bin/env python3
# %% imports
import logging
from pathlib import Path

from lib import save_for_R
from lib.data import (
    Column,
    make_all_data,
    select_variable,
    available_dummy_levels,
)

# ruff: noqa: F403, F405
from lib.data.variables import *
from lib.model import PanelModelDefinitionBuilder, VariableDefinition

logging.getLogger().setLevel(logging.DEBUG)

# %% Get data
all_data = make_all_data(cache=True)

# Drop rows for which the dependent variable is always NA, as these will never be included in a regression.
y = select_variable(all_data, MHI5)
y_missing = y.isna().sum(axis=1) == y.shape[1]
y_missing = y_missing[y_missing].index
all_data = all_data.drop(y_missing)

# %% model with single regression
model_definition = (
    PanelModelDefinitionBuilder()
    .with_y(VariableDefinition(MHI5))
    .with_x(VariableDefinition(SPORTS))
    .with_controls(
        [
            VariableDefinition(variable, dummy_levels=available_dummy_levels(all_data, variable))
            for variable in [
                AGE,
                ETHNICITY,
                GENDER,
                BMI,
                MARITAL_STATUS,
                INCOME,
                EDUCATION_LEVEL,
                EMPLOYMENT,
                PHYSICAL_HEALTH,
            ]
        ]
        + [VariableDefinition(variable) for variable in [PREVIOUS_DEPRESSION]]
    )
    .with_time_dummy()
    .with_excluded_regressors([Column(PREVIOUS_DEPRESSION, wave=13)])
    .with_dummy_level_covariances()
    .build(all_data)
)


print(model_definition)

# %% save for lavaan in R.
save_for_R(model_definition, all_data, Path("/tmp/panel_data.dta"))  # noqa: S108
