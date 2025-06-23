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
from lib.model import CovarianceDefinition, PanelModelDefinitionBuilder, VariableDefinition

logging.getLogger().setLevel(logging.INFO)

# %% Get data
all_data = make_all_data(cache=True)

# Drop rows for which the dependent variable is always NA, as these will never be included in a regression.
y = select_variable(all_data, MHI5)
y_missing = y.isna().sum(axis=1) == y.shape[1]
y_missing = y_missing[y_missing].index
all_data = all_data.drop(y_missing)

# Drop rows for which sports is always NA, as these provide no information towards our variable of interest
x = select_variable(all_data, SPORTS)
x_missing = x.isna().sum(axis=1) == x.shape[1]
x_missing = x_missing[x_missing].index
all_data = all_data.drop(x_missing)

# %% Base model (no mediators)
model_definition_base = (
    PanelModelDefinitionBuilder()
    .with_y(
        VariableDefinition(MHI5),
        lag_structure=[1, 2, 3],
    )
    .with_x(
        VariableDefinition(SPORTS),
        lag_structure=[0, 1],
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

print(model_definition_base)

# save for lavaan in R.
save_for_R(model_definition_base, all_data, Path("/tmp/panel_data.feather"))  # noqa: S108


# %% Model with mediators
model_definition_mediation = (
    PanelModelDefinitionBuilder()
    .with_y(
        VariableDefinition(MHI5),
        lag_structure=[1, 2, 3],
    )
    .with_x(
        VariableDefinition(SPORTS),
        lag_structure=[0, 1],
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
    .with_mediators(
        [
            VariableDefinition(variable, dummy_levels=available_dummy_levels(all_data, variable))
            for variable in [PHYSICAL_HEALTH, BMI]
        ]
        + [VariableDefinition(variable) for variable in [DISEASE_STATUS]]
    )
    .with_additional_covariances(
        fix_variance_across_time=False,
        free_covariance_across_time=True,
        within_dummy_covariance=True,  # False so that we can fix the parameter, ignoring "fix_variance_across_time"
        x_predetermined=False,
        # Manually define all the between-regressor covariances to give them a named parameter
        # The [1:]'s are to ignore the dummy levels left out for identification
        # between_regressors=[
        #     CovarianceDefinition(
        #         VariableDefinition(DISEASE_STATUS),
        #         [
        #             VariableDefinition(PHYSICAL_HEALTH, dummy_levels=available_dummy_levels(all_data, PHYSICAL_HEALTH)),
        #             VariableDefinition(BMI, dummy_levels=available_dummy_levels(all_data, BMI)),
        #         ],
        #     ),
        #     CovarianceDefinition(
        #         VariableDefinition(
        #             PHYSICAL_HEALTH, dummy_levels=available_dummy_levels(all_data, PHYSICAL_HEALTH)[1:2]
        #         ),
        #         [
        #             VariableDefinition(
        #                 PHYSICAL_HEALTH, dummy_levels=available_dummy_levels(all_data, PHYSICAL_HEALTH)[2:]
        #             ),
        #             VariableDefinition(BMI, dummy_levels=available_dummy_levels(all_data, BMI)[1:]),
        #         ],
        #     ),
        #     CovarianceDefinition(
        #         VariableDefinition(
        #             PHYSICAL_HEALTH, dummy_levels=available_dummy_levels(all_data, PHYSICAL_HEALTH)[2:3]
        #         ),
        #         [
        #             VariableDefinition(
        #                 PHYSICAL_HEALTH, dummy_levels=available_dummy_levels(all_data, PHYSICAL_HEALTH)[3:]
        #             ),
        #             VariableDefinition(BMI, dummy_levels=available_dummy_levels(all_data, BMI)[1:]),
        #         ],
        #     ),
        #     CovarianceDefinition(
        #         VariableDefinition(
        #             PHYSICAL_HEALTH, dummy_levels=available_dummy_levels(all_data, PHYSICAL_HEALTH)[3:4]
        #         ),
        #         [
        #             VariableDefinition(
        #                 PHYSICAL_HEALTH, dummy_levels=available_dummy_levels(all_data, PHYSICAL_HEALTH)[4:]
        #             ),
        #             VariableDefinition(BMI, dummy_levels=available_dummy_levels(all_data, BMI)[1:]),
        #         ],
        #     ),
        #     CovarianceDefinition(
        #         VariableDefinition(BMI, dummy_levels=available_dummy_levels(all_data, BMI)[1:2]),
        #         [VariableDefinition(BMI, dummy_levels=available_dummy_levels(all_data, BMI)[2:])],
        #     ),
        #     CovarianceDefinition(
        #         VariableDefinition(BMI, dummy_levels=available_dummy_levels(all_data, BMI)[2:3]),
        #         [VariableDefinition(BMI, dummy_levels=available_dummy_levels(all_data, BMI)[3:])],
        #     ),
        # ],
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

print(model_definition_mediation)

save_for_R(model_definition_mediation, all_data, Path("/tmp/panel_data.feather"))  # noqa: S108

# %% Models for cross-validation y
for max_lag in range(1, 8 + 1):
    model_definition_base = (
        PanelModelDefinitionBuilder()
        .with_y(
            VariableDefinition(MHI5),
            lag_structure=list(range(1, max_lag + 1)),
        )
        .with_x(
            VariableDefinition(SPORTS),
            lag_structure=[0, 1],
            fixed=True,
        )
        .with_controls(
            [
                VariableDefinition(variable, dummy_levels=available_dummy_levels(all_data, variable))
                for variable in [PHYSICAL_HEALTH]
            ]
            + [VariableDefinition(variable) for variable in [DISEASE_STATUS]]
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

    print("-" * 30)
    print(model_definition_base)
    print("-" * 30)

# %% Models for cross-validation x
for max_lag in range(1, 11 + 1):
    model_definition_base = (
        PanelModelDefinitionBuilder()
        .with_y(
            VariableDefinition(MHI5),
            lag_structure=[1, 2, 3],
        )
        .with_x(
            VariableDefinition(SPORTS),
            lag_structure=list(range(max_lag + 1)),
            fixed=True,
        )
        .with_controls(
            [
                VariableDefinition(variable, dummy_levels=available_dummy_levels(all_data, variable))
                for variable in [PHYSICAL_HEALTH]
            ]
            + [VariableDefinition(variable) for variable in [DISEASE_STATUS]]
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

    print("-" * 30)
    print(model_definition_base)
    print("-" * 30)
