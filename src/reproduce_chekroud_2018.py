#!/usr/bin/env python3
# %% imports
import logging
from pathlib import Path

import pandas as pd

from lib import save_for_R
from lib.data import (
    Column,
    make_all_data,
    select_variable,
    available_waves,
    select_wave,
    available_dummy_levels,
)

# ruff: noqa: F403, F405
from lib.data.variables import *
from lib.model import CSModelDefinitionBuilder, VariableDefinition

logging.getLogger().setLevel(logging.DEBUG)

# %% Get data
all_data = make_all_data(cache=True)

# Drop rows for which the dependent variable is always NA, as these will never be included in a regression.
y = select_variable(all_data, MHI5)
y_missing = y.isna().sum(axis=1) == y.shape[1]
y_missing = y_missing[y_missing].index
all_data = all_data.drop(y_missing)

# %% Flatten the data, lumping all years together in one big pile.
all_data_flattened = pd.DataFrame()


def remove_year(column: Column) -> Column:
    return Column(column.name, None, column.dummy_level)


for year in available_waves(all_data):
    subset = select_wave(all_data, year)

    columns: list[Column] = subset.columns  # pyright: ignore[reportAssignmentType]

    subset.columns = [remove_year(column) for column in columns]

    subset.index = pd.Index([f"{idx}_{year}" for idx in subset.index])

    all_data_flattened = pd.concat([all_data_flattened, subset])

# Drop missing dependent var
# These are left over from above missing_dependent_variable stuff, because that only removed variables when an
# individual was missing for all waves, but this deletes any waves that have missing y
y_missing_flat = all_data_flattened[Column(MHI5)].isna()
all_data_flattened = all_data_flattened.drop(y_missing_flat[y_missing_flat].index)

# %% model with single regression
model_definition = (
    CSModelDefinitionBuilder()
    .with_y(VariableDefinition(MHI5))
    .with_x(VariableDefinition(SPORTS))
    .with_controls(
        [
            VariableDefinition(variable, dummy_levels=available_dummy_levels(all_data_flattened, variable))
            for variable in [
                PHYSICAL_HEALTH,
                BMI,
                AGE,
                ETHNICITY,
                GENDER,
                MARITAL_STATUS,
                INCOME,
                EDUCATION_LEVEL,
                EMPLOYMENT,
            ]
        ]
        + [VariableDefinition(variable) for variable in [PREVIOUS_DEPRESSION]]
    )
    .build(all_data_flattened)
)


print(model_definition)

# %% save for lavaan in R.
save_for_R(model_definition, all_data_flattened, Path("/tmp/data.dta"))  # noqa: S108
