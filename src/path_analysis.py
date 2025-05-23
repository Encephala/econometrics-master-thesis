#!/usr/bin/env python3
# %% imports
import logging
from pathlib import Path

from lib import save_for_R
from lib.data import (
    make_all_data,
    select_variable,
    available_dummy_levels,
)

from lib.data import Column

# ruff: noqa: F403, F405
from lib.data.variables import *
from lib.model import PanelModelDefinitionBuilder, VariableDefinition

logging.getLogger().setLevel(logging.INFO)

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
    .with_y(
        VariableDefinition(MHI5),
        lag_structure=[1, 2, 3, 4],
    )
    .with_x(
        # VariableDefinition(SPORTS_WEEKLY_HOURS, dummy_levels=available_dummy_levels(all_data, SPORTS_WEEKLY_HOURS)),
        VariableDefinition(CUMULATIVE_SPORTS),
        lag_structure=[1],
        fixed=True,
    )
    .with_controls(
        [VariableDefinition(variable, dummy_levels=available_dummy_levels(all_data, variable)) for variable in []]
        + [VariableDefinition(variable) for variable in []]
        + [
            VariableDefinition(
                variable, is_time_invariant=True, dummy_levels=available_dummy_levels(all_data, variable)
            )
            for variable in [
                AGE,
                INCOME,
                ETHNICITY,
                GENDER,
                MARITAL_STATUS,
                EDUCATION_LEVEL,
                EMPLOYMENT,
            ]
        ]
        + [VariableDefinition(variable, is_time_invariant=True) for variable in []]
    )
    .with_additional_covariances(
        fix_variance_across_time=False,
        free_covariance_across_time=True,
        within_dummy_covariance=True,
        x_predetermined=False,
    )
    .with_excluded_regressors(
        [
            Column(f"{GENDER}_first", dummy_level="other"),  # Makes stuff unstable, it's only 10 True
            Column(f"{INCOME}_first", dummy_level="15k.50k"),  # Also unstable, only 9 True
        ]
    )
    .with_time_dummy()
    .build(all_data)
)


print(model_definition)

# %% save for lavaan in R.
save_for_R(model_definition, all_data, Path("/tmp/panel_data.feather"))  # noqa: S108


# %% Models for cross-validation
for max_lag in range(1, 11 + 1):
    model_definition = (
        PanelModelDefinitionBuilder()
        .with_y(
            VariableDefinition(MHI5),
            lag_structure=[1, 2, 3, 4],
        )
        .with_x(
            # VariableDefinition(SPORTS_WEEKLY_HOURS, dummy_levels=available_dummy_levels(all_data, SPORTS_WEEKLY_HOURS)),
            VariableDefinition(CUMULATIVE_SPORTS),
            lag_structure=list(range(1, max_lag + 1)),
            fixed=True,
        )
        .with_controls(
            [VariableDefinition(variable, dummy_levels=available_dummy_levels(all_data, variable)) for variable in []]
            + [VariableDefinition(variable) for variable in []]
            + [
                VariableDefinition(
                    variable, is_time_invariant=True, dummy_levels=available_dummy_levels(all_data, variable)
                )
                for variable in [
                    AGE,
                    INCOME,
                    ETHNICITY,
                    GENDER,
                    MARITAL_STATUS,
                    EDUCATION_LEVEL,
                    EMPLOYMENT,
                ]
            ]
            + [VariableDefinition(variable, is_time_invariant=True) for variable in []]
        )
        .with_additional_covariances(
            fix_variance_across_time=False,
            free_covariance_across_time=True,
            within_dummy_covariance=True,
            x_predetermined=False,
        )
        .with_excluded_regressors(
            [
                Column(f"{GENDER}_first", dummy_level="other"),  # Makes stuff unstable, it's only 10 True
                Column(f"{INCOME}_first", dummy_level="15k.50k"),  # Also unstable, only 9 True
            ]
        )
        .with_time_dummy()
        .with_excluded_regressand_waves(list(range(8, 23)))
        .build(all_data)
    )

    print(model_definition)
