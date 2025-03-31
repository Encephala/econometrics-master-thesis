#!/usr/bin/env python3
# %% imports
import pandas as pd
import numpy as np

import semopy

from util.data import (
    Column,
    load_wide_panel_cached,
    select_variable,
    select_wave,
    available_years,
    available_dummy_levels,
    cleanup_dummy,
    map_columns_to_str,
)
from util.model import ModelDefinitionBuilder, VariableDefinition


# %% data loading
background_vars = load_wide_panel_cached("avars")
leisure_panel = load_wide_panel_cached("cs")
health_panel = load_wide_panel_cached("ch")

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
unhappy = select_variable(health_panel, UNHAPPY)

mapper = {
    "never": 0,
    "seldom": 1,
    "sometimes": 3,
    "often": 10,
    "mostly": 16,
    "continuously": 30,
}

unhappy = unhappy.apply(lambda column: column.map(mapper, na_action="ignore"))

# %% Making sports an actual boolean instead of "yes"/"no"
sports = select_variable(leisure_panel, SPORTS)

sports = sports.apply(lambda column: column.map({"yes": True, "no": False}, na_action="ignore")).astype("boolean")

# %% age preprocessing
# From Chekroud 2018
age = select_variable(background_vars, AGE)


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

for column in age:
    new_column = pd.cut(
        age[column],
        bins=[-np.inf, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, np.inf],
        labels=age_labels,
    )

    age = age.copy()
    age[column] = new_column

# %% Income preprocessing
# From Chekroud 2018
income = select_variable(background_vars, INCOME)

income_labels = [
    "under 15000",
    "15 to 25000",
    "25 to 35000",
    "35 to 50000",
    "over 50000",
]

for column in income:
    new_column = pd.cut(
        income[column],
        bins=[-np.inf, 15000, 25000, 35000, 50000, np.inf],
        labels=income_labels,
    )

    income = income.copy()
    income[column] = new_column

# %% derive employment from primary occupation
occupation = select_variable(background_vars, PRINCIPAL_OCCUPATION)

EMPLOYMENT = "employment"


def merge_and_map_categories(column: pd.Series) -> pd.Series:
    EMPLOYED = "employed"
    SELF_EMPLOYED = "self-employed"
    OUT_OF_WORK = "out of work"
    HOMEMAKER = "homemaker"
    STUDENT = "student"
    RETIRED = "retired"
    UNABLE = "unable to work"

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

    # Apparently if you set name=... here, it gets ignored because of .apply
    return pd.Series(result, index=column.index)


# Apply column-wise to have cohesive datatype
employment = occupation.apply(merge_and_map_categories)

columns: list[Column] = employment.columns  # pyright: ignore[reportAssignmentType]
employment.columns = [Column(EMPLOYMENT, column.wave) for column in columns]

# %% calculate BMI
weight = select_variable(health_panel, WEIGHT)
height = select_variable(health_panel, HEIGHT)

# Broad sanity check
TALLEST_HEIGHT_EVER = 270  # According to google idk
VERY_SHORT_TEEN = 100
height = height.mask(height > TALLEST_HEIGHT_EVER, pd.NA)
height = height.mask(height < VERY_SHORT_TEEN, pd.NA)

HEAVIEST_PERSON_EVER = 635  # According to google
VERY_LIGHT_TEEN = 30
weight = weight.mask(weight > HEAVIEST_PERSON_EVER, pd.NA)
weight = weight.mask(weight < VERY_LIGHT_TEEN, pd.NA)

bmi = pd.DataFrame(index=health_panel.index)

BMI = "bmi"

for year in available_years(weight):  # Can choose either weight or height, if one is missing answer is NA anyways
    bmi[Column(BMI, year)] = weight[Column(WEIGHT, year)] / (height[Column(HEIGHT, year)] / 100) ** 2

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
depression = select_variable(health_panel, DEPRESSION_MEDICATION)

years_depression = sorted(available_years(depression))
names_depression = [Column(DEPRESSION_MEDICATION, year) for year in sorted(available_years(depression))]

previous_depression = pd.DataFrame(index=health_panel.index)
PREVIOUS_DEPRESSION = "prev_depr"

for person in previous_depression.index:
    medication_status: pd.Series = depression.loc[person, names_depression].squeeze()

    medication_status = medication_status.map(lambda x: x == "yes", na_action="ignore")

    cumulative_medication_status = pd.NA
    for year in years_depression[1:]:
        previous_year_name = Column(DEPRESSION_MEDICATION, year - 1)

        is_previous_year_available = pd.isna(medication_status.get(previous_year_name))

        if not is_previous_year_available:
            if pd.isna(cumulative_medication_status):
                cumulative_medication_status = medication_status[previous_year_name]  # pyright: ignore  # noqa: PGH003
            else:
                cumulative_medication_status = cumulative_medication_status or medication_status[previous_year_name]  # pyright: ignore  # noqa: PGH003

        previous_depression.loc[person, Column(PREVIOUS_DEPRESSION, year)] = cumulative_medication_status

# Fix dtype, pandas doesn't automatically recognise a combination of NA and bool is just nullable boolean
previous_depression = previous_depression.astype("boolean")

# %% Make education level more sane
education = select_variable(background_vars, EDUCATION_LEVEL)

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
ethnicity = select_variable(background_vars, ETHNICITY)

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

DUMMY_NA = True
DROP_FIRST = False


def make_dummies(df: pd.DataFrame) -> pd.DataFrame:
    # NOTE: Drop first for identification or use NA level for identification
    result = pd.get_dummies(df, prefix_sep=".", dummy_na=True, drop_first=True)

    new_columns: list[Column] = []
    for column in result.columns:
        name = column[: column.rfind("_")]
        wave = int(column[column.rfind("_") + 1 : column.find(".")])
        dummy_level = column[column.find(".") + 1 :]

        new_columns.append(Column(name, wave, cleanup_dummy(dummy_level)))

    result.columns = new_columns

    return result


all_relevant_data = pd.DataFrame(index=background_vars.index).join(
    [
        pd.Series(1, index=background_vars.index, name=Column(CONSTANT)),
        unhappy,
        sports,
        make_dummies(age),
        make_dummies(ethnicity),
        make_dummies(select_variable(background_vars, GENDER)),
        make_dummies(select_variable(background_vars, MARITAL_STATUS)),
        make_dummies(income),
        make_dummies(education),
        make_dummies(employment),
        make_dummies(select_variable(health_panel, PHYSICAL_HEALTH)),
        make_dummies(bmi),
        previous_depression,
    ]
)

# Drop rows for which the dependent variable is always NA, as these will never be included in a regression.
missing_dependent_variable = select_variable(all_relevant_data, UNHAPPY)
missing_dependent_variable = missing_dependent_variable.isna().sum(axis=1) == missing_dependent_variable.shape[1]
missing_dependent_variable_index = missing_dependent_variable[missing_dependent_variable].index
all_relevant_data = all_relevant_data.drop(missing_dependent_variable_index)

# Sort columns
all_relevant_data = all_relevant_data[sorted(all_relevant_data.columns)]

# %% Flatten the data, lumping all years together in one big pile.
all_data_flattened = pd.DataFrame()


def remove_year(column: Column) -> Column:
    return Column(column.name, None, column.dummy_level)


for year in available_years(all_relevant_data):
    subset = select_wave(all_relevant_data, year)

    columns: list[Column] = subset.columns  # pyright: ignore[reportAssignmentType]

    subset.columns = [remove_year(column) for column in columns]

    subset.index = pd.Index([f"{idx}_{year}" for idx in subset.index])

    all_data_flattened = pd.concat([all_data_flattened, subset])

# %% model with single regression
model_single_regression = (
    ModelDefinitionBuilder()
    .with_y(VariableDefinition(UNHAPPY))
    .with_x(VariableDefinition(SPORTS))
    .with_w(
        [
            VariableDefinition(variable, dummy_levels=available_dummy_levels(all_data_flattened, variable))
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
    )
    .with_excluded_regressors([Column(AGE, None, dummy_level="nan")])
    .build_nonpanel(all_data_flattened)
)


print(model_single_regression)

model = semopy.Model(model_single_regression)

# TODO: Drop `geslacht.nan` as a regressor because it is highly colinear with `leeftijd.nan`.

# %% save for lavaan in R.
# Relevant for now, because obj="FIML" gives invalid value warnings in semopy
all_data_flattened.astype("float64").to_stata("/tmp/data.dta")  # noqa: S108

print("Model definition in stata/lavaan form:")
print(model_single_regression.replace(".", "_"))

# %% fit that one
data_flattened = map_columns_to_str(all_data_flattened.astype(np.float64))
optimisation_result = model.fit(data_flattened)

print(optimisation_result)

model.inspect().sort_values(["op", "rval"])  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

# %% naive model definition
model_definition = (
    ModelDefinitionBuilder()
    .with_y(VariableDefinition(UNHAPPY))
    .with_x(VariableDefinition(SPORTS))
    .with_w(
        [
            VariableDefinition(variable, dummy_levels=available_dummy_levels(all_relevant_data, variable))
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
    )
    .build(all_relevant_data)
)

print(model_definition)

model = semopy.Model(model_definition)

# %% naive model
all_data = map_columns_to_str(all_relevant_data.astype(np.float64))
optimisation_result = model.fit(all_data, obj="FIML")

print(optimisation_result)

model.inspect().sort_values(["op", "Estimate", "lval"])  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
