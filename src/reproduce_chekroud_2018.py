#!/usr/bin/env python3
# %% imports

from util.data import load_wide_panel_cached, standardise_wide_column_name

# %% data loading
background_vars = load_wide_panel_cached("avars").rename(columns=standardise_wide_column_name)
leisure_panel = load_wide_panel_cached("cs").rename(columns=standardise_wide_column_name)
health_panel = load_wide_panel_cached("ch").rename(columns=standardise_wide_column_name)

# %% select variables
UNHAPPY = "ch14"
SPORTS = "cs104"

# race, gender, marital status, income, education level, BMI (category), (self-reported) physical health and previous diagnosis of depression
AGE = "leeftijd"  # TODO: How is this included? stratified?
GENDER = "geslacht"  # TODO: Why is this only availabel for 13k out of 31k people?
RACE = "herkomstgroep"  # TODO: This one requires some pre-processing
