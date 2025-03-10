#!/usr/bin/env python3
# %% imports
import pandas as pd
import numpy as np

from util.data import (
    load_wide_panel_cached,
    standardise_wide_column_name,
    select_question_wide,
    available_years,
)


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
EDUCATION_LEVEL = "oplcat"  # TODO: What exactly in Chekroud?
EMPLOYMENT_STATUS = ...  # TODO
PHYSICAL_HEALTH = "ch4"  # TODO: dummies
# For BMI
HEIGHT = "ch16"
WEIGHT = "ch17"
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

age = select_question_wide(background_vars, AGE)

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

income = select_question_wide(background_vars, INCOME)

for column in income:
    new_column = pd.cut(income[column], bins=income_cutoffs)
    new_column = new_column.cat.rename_categories(income_labels)

    income = income.assign(**{column: new_column})  # pyright: ignore[reportCallIssue]


# %% calculate BMI
weight = select_question_wide(health_panel, WEIGHT)
height = select_question_wide(health_panel, HEIGHT)

bmi = pd.DataFrame(index=health_panel.index)

for year in available_years(weight):  # Can choose either weight or height, if one is missing answer is NA anyways
    bmi[f"bmi_{year}"] = weight[f"{WEIGHT}_{year}"] / (height[f"{HEIGHT}_{year}"] / 100) ** 2

# TODO: There's some excessive BMI values, need to filter
# Also need to then stratify (pd.cut)

# %% determine previous depression status
depression = select_question_wide(health_panel, DEPRESSION_MEDICATION)

years_depression = sorted(available_years(depression))
names_depression = [f"{DEPRESSION_MEDICATION}_{year}" for year in sorted(available_years(depression))]

previous_depression = pd.DataFrame(index=health_panel.index)

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

        previous_depression.loc[person, f"prev_depr_{year}"] = cumulative_medication_status

# %% the big merge
all_relevant_data = pd.DataFrame(index=background_vars.index).join(
    [
        age,
        select_question_wide(background_vars, RACE),
        select_question_wide(background_vars, GENDER),
        select_question_wide(background_vars, MARITAL_STATUS),
        income,
        select_question_wide(background_vars, EDUCATION_LEVEL),
        select_question_wide(health_panel, PHYSICAL_HEALTH),
        select_question_wide(health_panel, UNHAPPY),
        select_question_wide(leisure_panel, SPORTS),
        bmi,
        previous_depression,
    ]
)
