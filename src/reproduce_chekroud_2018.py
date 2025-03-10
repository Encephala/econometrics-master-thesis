#!/usr/bin/env python3
# %% imports
import pandas as pd
import numpy as np

from util.data import (
    load_wide_panel_cached,
    standardise_wide_column_name,
    select_question_wide,
    available_years,
    fix_column_categories,
)


# %% data loading
background_vars = load_wide_panel_cached("avars").rename(columns=standardise_wide_column_name)
leisure_panel = load_wide_panel_cached("cs").rename(columns=standardise_wide_column_name)
health_panel = load_wide_panel_cached("ch").rename(columns=standardise_wide_column_name)

# %% all variable names
UNHAPPY = "ch14"
SPORTS = "cs104"

AGE = "leeftijd"  # TODO: How is this included in Chekroud's model? stratified?
RACE = "herkomstgroep"  # TODO: This one requires some pre-processing, dummies?
GENDER = "geslacht"  # TODO: Why is this only availabel for 13k out of 31k people?
MARITAL_STATUS = "burgstat"  # TODO: dummies - perhaps "partner" is better, compare to Chekroud
INCOME = "nettohh_f"  # TODO: is Chekroud bruto? And household or individiual?
EDUCATION_LEVEL = "oplcat"  # TODO: What exactly in Chekroud?
PHYSICAL_HEALTH = "ch4"  # TODO: dummies
# For BMI
# TODO: Derive BMI
HEIGHT = "ch16"
WEIGHT = "ch17"
DEPRESSION_MEDICATION = "ch178"  # TODO: convert to previous diagnosis (taking medication implies diagnosis) (wack, that won't be easy with wide panel)

# %% calculate BMI
weight = select_question_wide(health_panel, WEIGHT)
height = select_question_wide(health_panel, HEIGHT)

bmi = pd.DataFrame(index=health_panel.index)

for year in available_years(weight):  # Can choose either weight or height, if one is missing answer is NA anyways
    bmi[f"bmi_{year}"] = weight[f"{WEIGHT}_{year}"] / (height[f"{HEIGHT}_{year}"] / 100) ** 2

# TODO: There's some excessive BMI values, need to filter

# %% determine previous depression status
depression = select_question_wide(health_panel, DEPRESSION_MEDICATION)

years_depression = sorted(available_years(depression))
names_depression = [f"{DEPRESSION_MEDICATION}_{year}" for year in sorted(available_years(depression))]

previous_depression = pd.DataFrame(index=health_panel.index)

for person in previous_depression.index:
    medication_status: "pd.Series[bool]" = depression.loc[person, names_depression].squeeze()

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
        select_question_wide(background_vars, AGE),
        select_question_wide(background_vars, RACE),
        select_question_wide(background_vars, GENDER),
        select_question_wide(background_vars, MARITAL_STATUS),
        select_question_wide(background_vars, INCOME),
        select_question_wide(background_vars, EDUCATION_LEVEL),
        select_question_wide(health_panel, PHYSICAL_HEALTH),
        select_question_wide(health_panel, UNHAPPY),
        select_question_wide(leisure_panel, SPORTS),
        bmi,
        previous_depression,
    ]
)
