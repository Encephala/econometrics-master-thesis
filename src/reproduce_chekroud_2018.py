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
from util.data.processing import cleanup_dummy, cleanup_dummy_column
from util.model import ModelDefinitionBuilder, VariableDefinition


# %% data loading
background_vars = load_wide_panel_cached("avars")
leisure_panel = load_wide_panel_cached("cs").rename(columns=standardise_wide_column_name)
health_panel = load_wide_panel_cached("ch").rename(columns=standardise_wide_column_name)

# %% all variable names
UNHAPPY = "ch14"
SPORTS = "cs104"

AGE = "leeftijd"
ETHNICITY = "herkomstgroep"
GENDER = "geslacht"
MARITAL_STATUS = "burgstat"
INCOME = "nettohh_f"
EDUCATION_LEVEL = "oplcat"
PRINCIPAL_OCCUPATION = "belbezig"  # To derive employment status
PHYSICAL_HEALTH = "ch4"
HEIGHT, WEIGHT = "ch16", "ch17"  # For BMI
DEPRESSION_MEDICATION = "ch178"

# %% Converting unhappiness categories to numbers
unhappy = select_variable_wide(health_panel, UNHAPPY)

mapper = {
    "never": 0,
    "seldom": 1,
    "sometimes": 2,
    "often": 3,
    "mostly": 4,
    "continuously": 5,
}

unhappy = unhappy.apply(lambda column: column.map(mapper, na_action="ignore")).astype(np.float64)

# %% Making sports an actual boolean instead of "yes"/"no"
sports = select_variable_wide(leisure_panel, SPORTS)

sports = sports.apply(lambda column: column.map({"yes": True, "no": False}, na_action="ignore")).astype("boolean")

# %% age preprocessing
# From Chekroud 2018
age_cutoffs = pd.IntervalIndex.from_breaks(
    [-np.inf, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, np.inf],
    closed="left",
)

age_labels = [
    "under 18",
    "18 to 24 years",
    "25 to 29 years",
    "30 to 34 years",
    "35 to 39 years",
    "40 to 44 years",
    "45 to 49 years",
    "50 to 54 years",
    "55 to 59 years",
    "60 to 64 years",
    "65 to 69 years",
    "70 to 74 years",
    "75 to 79 years",
    "over 80",
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
    "under 15000",
    "15 to 25000",
    "25 to 35000",
    "35 to 50000",
    "over 50000",
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
    EMPLOYED = "employed"
    SELF_EMPLOYED = "self-employed"
    OUT_OF_WORK = "out of work"
    HOMEMAKER = "homemaker"
    STUDENT = "student"
    RETIRED = "retired"
    UNABLE = "unable to work"

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

    return pd.Series(result, name=f"{EMPLOYMENT}_{year}", index=column.index)


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

# Broad sanity check
TALLEST_HEIGHT_EVER = 270  # According to google idk
SHORT_TEEN = 120
height = height.mask(height > TALLEST_HEIGHT_EVER, pd.NA)
height = height.mask(height < SHORT_TEEN, pd.NA)

HEAVIEST_PERSON_EVER = 635  # According to google
LIGHT_TEEN = 40
weight = weight.mask(weight > HEAVIEST_PERSON_EVER, pd.NA)
weight = weight.mask(weight < LIGHT_TEEN, pd.NA)

bmi = pd.DataFrame(index=health_panel.index)

BMI = "bmi"

for year in available_years(weight):  # Can choose either weight or height, if one is missing answer is NA anyways
    bmi[f"{BMI}_{year}"] = weight[f"{WEIGHT}_{year}"] / (height[f"{HEIGHT}_{year}"] / 100) ** 2

# BMI ranges from https://www.nhs.uk/conditions/obesity/
bmi = bmi.apply(
    lambda column: pd.cut(
        column,
        bins=[-np.inf, 18.5, 25.0, 30, np.inf],
        labels=["underweight", "normal weight", "overweight", "obese"],
        right=False,
    )
)

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

# %% Make education level more sane
education = select_variable_wide(background_vars, EDUCATION_LEVEL)

category_map = {
    "havo/vwo (higher secondary education/preparatory university education, us: senior high school)": "havo vwo",
    "hbo (higher vocational education, us: college)": "hbo",
    "mbo (intermediate vocational education, us: junior college)": "mbo",
    "primary school": "primary school",
    "vmbo (intermediate secondary education, us: junior high school)": "vmbo",
    "wo (university)": "wo",
}

education = education.apply(lambda column: column.cat.rename_categories(category_map))

# %% Make ethnicity more sane
ethnicity = select_variable_wide(background_vars, ETHNICITY)

category_map = {
    "dutch background": "dutch",
    "first generation foreign, non-western background": "first nonw",
    "first generation foreign, western background": "first w",
    "second generation foreign, non-western background": "second nonw",
    "second generation foreign, western background": "second w",
}

ethnicity = ethnicity.apply(lambda column: column.cat.rename_categories(category_map))

# %% the big merge
CONSTANT = "constant"

# TODO: Remove all prefixes from category names somewhere in the code (probably loading, not here?) ( :^) )
# NOTE: Use | as dummy separator to not conflict with <question>_<year>, drop first for identification
all_relevant_data = pd.DataFrame(index=background_vars.index).join(
    [
        pd.Series(1, index=background_vars.index, name=CONSTANT),
        unhappy,
        sports,
        pd.get_dummies(age, prefix_sep=".", dummy_na=True, drop_first=True),
        pd.get_dummies(ethnicity, prefix_sep=".", dummy_na=True, drop_first=True),
        pd.get_dummies(select_variable_wide(background_vars, GENDER), prefix_sep=".", dummy_na=True, drop_first=True),
        pd.get_dummies(
            select_variable_wide(background_vars, MARITAL_STATUS), prefix_sep=".", dummy_na=True, drop_first=True
        ),
        pd.get_dummies(income, prefix_sep=".", dummy_na=True, drop_first=True),
        pd.get_dummies(education, prefix_sep=".", dummy_na=True, drop_first=True),
        pd.get_dummies(employment, prefix_sep=".", dummy_na=True, drop_first=True),
        pd.get_dummies(
            select_variable_wide(health_panel, PHYSICAL_HEALTH), prefix_sep=".", dummy_na=True, drop_first=True
        ),
        pd.get_dummies(bmi, prefix_sep=".", dummy_na=True, drop_first=True),
        previous_depression,
    ]
)

# Sort columns
all_relevant_data = all_relevant_data[sorted(all_relevant_data.columns)]

# Standardise names of dummy levels
all_relevant_data = all_relevant_data.rename(columns=cleanup_dummy_column)

# %% naive model definition
model_definition = (
    ModelDefinitionBuilder()
    .with_y(VariableDefinition(UNHAPPY))
    .with_x(VariableDefinition(SPORTS))
    .with_w(
        [
            VariableDefinition(
                variable,
                dummy_levels=[cleanup_dummy(level) for level in available_dummy_levels(all_relevant_data, variable)],
            )
            for variable in [
                AGE,
                ETHNICITY,
                GENDER,
                MARITAL_STATUS,
                INCOME,
                EDUCATION_LEVEL,
                EMPLOYMENT,
                PHYSICAL_HEALTH,
                BMI,
            ]
        ]
        + [VariableDefinition(variable) for variable in [PREVIOUS_DEPRESSION]]
    )  # All are dummies for now
    .build(all_relevant_data)
)

print(model_definition)

model = semopy.ModelMeans(model_definition)

# %% naive model
optimisation_result = model.fit(all_relevant_data.astype(np.float64))

print(optimisation_result)

model.inspect().sort_values(["op", "Estimate", "lval"])  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

# %% model doesn't work in python, saving for R.
all_relevant_data.astype("float64").to_stata("/tmp/data.dta")  # noqa: S108

print("Model definition in stata/lavaan form:")
print(model_definition.replace(".", "_"))
