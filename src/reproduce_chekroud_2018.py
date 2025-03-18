#!/usr/bin/env python3
# %% imports
import pandas as pd
import numpy as np

import semopy

from util.data import (
    load_wide_panel_cached,
    standardise_wide_column_name,
    select_variable_wide,
    available_years,
    available_dummy_levels,
)
from util.model import ModelDefinitionBuilder, VariableDefinition


# %% data loading
background_vars = load_wide_panel_cached("avars")
leisure_panel = load_wide_panel_cached("cs").rename(columns=standardise_wide_column_name)
health_panel = load_wide_panel_cached("ch").rename(columns=standardise_wide_column_name)

# %% all variable names
UNHAPPY = "ch14"
SPORTS = "cs104"

AGE = "leeftijd"
RACE = "herkomstgroep"
GENDER = "geslacht"
MARITAL_STATUS = "burgstat"
INCOME = "nettohh_f"
EDUCATION_LEVEL = "oplcat"
PRINCIPAL_OCCUPATION = "belbezig"  # To derive employment status
PHYSICAL_HEALTH = "ch4"
HEIGHT, WEIGHT = "ch16", "ch17"  # For BMI
DEPRESSION_MEDICATION = "ch178"

# %% age preprocessing
# From Chekroud 2018
age_cutoffs = pd.IntervalIndex.from_breaks(
    [-np.inf, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, np.inf],
    closed="left",
)

age_labels = [
    "18-",
    "18-24 years",
    "25-29 years",
    "30-34 years",
    "35-39 years",
    "40-44 years",
    "45-49 years",
    "50-54 years",
    "55-59 years",
    "60-64 years",
    "65-69 years",
    "70-74 years",
    "75-79 years",
    "80+",
]

age = select_variable_wide(background_vars, AGE)

for column in age:
    new_column = pd.cut(age[column], bins=age_cutoffs)
    new_column = new_column.cat.rename_categories(age_labels)

    age = age.assign(**{column: new_column})  # pyright: ignore[reportCallIssue]

# %% Income preprocessing
# From Chekroud 2018
income_cutoffs = pd.IntervalIndex.from_breaks(
    [-np.inf, 15000, 25000, 35000, 50000, np.inf],
    closed="left",
)

income_labels = [
    "< € 15.000",
    "€ 15-25.000",
    "€ 25-35.000",
    "€ 35-50.000",
    "> € 50.000",
]

income = select_variable_wide(background_vars, INCOME)

for column in income:
    new_column = pd.cut(income[column], bins=income_cutoffs)
    new_column = new_column.cat.rename_categories(income_labels)

    income = income.assign(**{column: new_column})  # pyright: ignore[reportCallIssue]

# %% derive employment from primary occupation
occupation = select_variable_wide(background_vars, PRINCIPAL_OCCUPATION)

EMPLOYMENT = "employment"


def merge_and_map_categories(column: pd.Series) -> pd.Series:
    EMPLOYED = "Employed"
    SELF_EMPLOYED = "Self-employed"
    OUT_OF_WORK = "Out of work"
    HOMEMAKER = "Homemaker"
    STUDENT = "Student"
    RETIRED = "Retired"
    UNABLE = "Unable to work"

    year = column.name[column.name.rfind("_") + 1 :]  # pyright: ignore # noqa: PGH003

    # Map to new codes
    old_category_to_new_category = {
        "paid employment": EMPLOYED,
        "works or assists in family business": EMPLOYED,
        "autonomous professional, freelancer, or self-employed": SELF_EMPLOYED,
        "job seeker following job loss": OUT_OF_WORK,
        "first-time job seeker": OUT_OF_WORK,
        "exempted from job seeking following job loss": UNABLE,
        "attends school or is studying": STUDENT,
        "takes care of the housekeeping": HOMEMAKER,
        "is pensioner ([voluntary] early retirement, old age pension scheme)": RETIRED,
        "has (partial) work disability": UNABLE,
        # Not sure about all the ones below, think they're the best?
        "performs unpaid work while retaining unemployment benefit": EMPLOYED,
        "performs voluntary work": EMPLOYED,
        "does something else": EMPLOYED,
        "is too young to have an occupation": STUDENT,
    }

    result = pd.Categorical(column.map(old_category_to_new_category))

    return pd.Series(result, name=f"{EMPLOYMENT}_{year}")


# Apply column-wise to have cohesive datatype
employment = occupation.apply(merge_and_map_categories)
new_column_names = []
for column in employment.columns:
    year = column[column.rfind("_") + 1 :]

    new_column_names.append(f"{EMPLOYMENT}_{year}")
employment.columns = new_column_names

# %% calculate BMI
weight = select_variable_wide(health_panel, WEIGHT)
height = select_variable_wide(health_panel, HEIGHT)

bmi = pd.DataFrame(index=health_panel.index)

BMI = "bmi"

for year in available_years(weight):  # Can choose either weight or height, if one is missing answer is NA anyways
    bmi[f"{BMI}_{year}"] = weight[f"{WEIGHT}_{year}"] / (height[f"{HEIGHT}_{year}"] / 100) ** 2

# TODO: There's some excessive BMI values, need to filter
# Also need to then stratify (pd.cut)

# %% determine previous depression status
depression = select_variable_wide(health_panel, DEPRESSION_MEDICATION)

years_depression = sorted(available_years(depression))
names_depression = [f"{DEPRESSION_MEDICATION}_{year}" for year in sorted(available_years(depression))]

previous_depression = pd.DataFrame(index=health_panel.index)
PREVIOUS_DEPRESSION = "prev_depr"

for person in previous_depression.index:
    medication_status: pd.Series = depression.loc[person, names_depression].squeeze()

    medication_status = medication_status.map(lambda x: x == "yes", na_action="ignore")

    cumulative_medication_status = pd.NA
    for year in years_depression[1:]:
        previous_year_name = f"{DEPRESSION_MEDICATION}_{year - 1}"

        is_previous_year_available = pd.isna(medication_status.get(previous_year_name))

        if not is_previous_year_available:
            if pd.isna(cumulative_medication_status):
                cumulative_medication_status = medication_status[previous_year_name]
            else:
                cumulative_medication_status = cumulative_medication_status or medication_status[previous_year_name]

        previous_depression.loc[person, f"{PREVIOUS_DEPRESSION}_{year}"] = cumulative_medication_status

# Fix dtype, pandas doesn't automatically recognise a combination of NA and bool is just nullable boolean
previous_depression = previous_depression.astype("boolean")

# %% the big merge
CONSTANT = "const"

# TODO: Remove all prefixes from category names somewhere in the code (probably loading, not here?) ( :^) )
# NOTE: Use | as dummy separator to not conflict with <question>_<year>, drop first for identification
all_relevant_data = pd.DataFrame(index=background_vars.index).join(
    [
        pd.Series(1, index=background_vars.index, name=CONSTANT),
        select_variable_wide(health_panel, UNHAPPY),
        select_variable_wide(leisure_panel, SPORTS),
        pd.get_dummies(age, prefix_sep="|", dummy_na=True, drop_first=True),
        pd.get_dummies(select_variable_wide(background_vars, RACE), prefix_sep="|", dummy_na=True, drop_first=True),
        pd.get_dummies(select_variable_wide(background_vars, GENDER), prefix_sep="|", dummy_na=True, drop_first=True),
        pd.get_dummies(
            select_variable_wide(background_vars, MARITAL_STATUS), prefix_sep="|", dummy_na=True, drop_first=True
        ),
        pd.get_dummies(income, prefix_sep="|", dummy_na=True, drop_first=True),
        pd.get_dummies(
            select_variable_wide(background_vars, EDUCATION_LEVEL), prefix_sep="|", dummy_na=True, drop_first=True
        ),
        pd.get_dummies(employment, prefix_sep="|", dummy_na=True, drop_first=True),
        pd.get_dummies(
            select_variable_wide(health_panel, PHYSICAL_HEALTH), prefix_sep="|", dummy_na=True, drop_first=True
        ),
        bmi,
        previous_depression,
    ]
)

# Sort columns
all_relevant_data = all_relevant_data[sorted(all_relevant_data.columns)]

all_controls = [
    AGE,
    RACE,
    GENDER,
    MARITAL_STATUS,
    INCOME,
    EDUCATION_LEVEL,
    EMPLOYMENT,
    PHYSICAL_HEALTH,
    BMI,
    PREVIOUS_DEPRESSION,
]

# %% naive model definition
model_definition = (
    ModelDefinitionBuilder()
    .with_y(VariableDefinition(UNHAPPY))
    .with_x(VariableDefinition(SPORTS))
    .with_constant(VariableDefinition(CONSTANT))
    .with_w(
        [
            VariableDefinition(variable, dummy_levels=available_dummy_levels(all_relevant_data, variable))
            for variable in [AGE, RACE, GENDER, MARITAL_STATUS, INCOME, EDUCATION_LEVEL, EMPLOYMENT, PHYSICAL_HEALTH]
        ]
        + [VariableDefinition(variable) for variable in [BMI, PREVIOUS_DEPRESSION]]
    )  # All are dummies for now
    .build(all_relevant_data.columns)
)

print(model_definition)

model = semopy.Model(model_definition)  # %% naive model

# %% naive model
optimisation_result = model.fit(all_relevant_data)

print(optimisation_result)

model.inspect().sort_values(["op", "Estimate", "lval"])  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
