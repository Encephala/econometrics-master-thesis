from dataclasses import replace
from pathlib import Path
import logging

import pandas as pd
import numpy as np

# ruff: noqa: F403, F405
from .loading import Column, load_wide_panel_cached
from .util import assert_column_type_correct, available_years, select_variable, cleanup_dummy
from .variables import *

logger = logging.getLogger(__name__)


# All variable names
# Dependent variable
def map_mhi5_categories(series: pd.Series, *, is_positive: bool = False) -> pd.Series:
    "Takes a Categorical series of MHI-5 questionnaire responses and maps the textual responses to int values."
    # https://www.cbs.nl/nl-nl/achtergrond/2015/18/beperkingen-in-dagelijkse-handelingen-bij-ouderen/mhi-5
    result = series.map(
        {
            "continuously": 0,
            "mostly": 1,
            "often": 2,
            "sometimes": 3,
            "seldom": 4,
            "never": 5,
        },
        na_action="ignore",
    ).astype("Int64")

    if is_positive:
        result = 5 - result

    return result


def make_mhi5(health_panel: pd.DataFrame) -> pd.DataFrame:
    """Calculates the MHI-5 score for each individual-year available in the passed dataframe.

    Returns the MHI-5 scores in a new df (does not mutate the passed df)."""
    assert_column_type_correct(health_panel)

    # Relevant question indices
    ANXIOUS = "ch11"
    CANT_CHEER_UP = "ch12"
    CALM_PEACEFUL = "ch13"
    DEPRESSED_GLOOMY = "ch14"
    HAPPY = "ch15"

    # The epic calculation
    result = pd.DataFrame()

    for year in available_years(health_panel):
        # These contain the responses of all individuals for this `year`
        anxious = map_mhi5_categories(health_panel[Column(ANXIOUS, year)])
        cheer_up = map_mhi5_categories(health_panel[Column(CANT_CHEER_UP, year)])
        calm = map_mhi5_categories(health_panel[Column(CALM_PEACEFUL, year)], is_positive=True)
        depressed = map_mhi5_categories(health_panel[Column(DEPRESSED_GLOOMY, year)])
        happy = map_mhi5_categories(health_panel[Column(HAPPY, year)], is_positive=True)

        # NOTE: There is no partial missingness, as in whenever one variable is missing,
        # all are missing. No need to f.i. impute NAs as the average of the other variables
        mhi = 4 * (anxious + cheer_up + calm + depressed + happy)

        result[Column("mhi5", year)] = mhi

    return result


# Sports
def make_sports(leisure_panel: pd.DataFrame) -> pd.DataFrame:
    # Making it an actual boolean instead of "yes"/"no"
    sports = select_variable(leisure_panel, SPORTS)

    return sports.apply(lambda column: column.map({"yes": True, "no": False}, na_action="ignore")).astype("boolean")


# Age
def make_age(background_vars: pd.DataFrame) -> pd.DataFrame:
    # From Chekroud 2018
    age = select_variable(background_vars, AGE)

    age_labels = [
        "under 18",
        "18-24",
        "25-39",
        "40-66",
        "over 67",
    ]

    return age.apply(
        lambda column: pd.cut(
            column,
            bins=[-np.inf, 17, 24, 39, 66, np.inf],
            labels=age_labels,
        )
    )


# Income
def make_income(background_vars: pd.DataFrame) -> pd.DataFrame:
    # From Chekroud 2018
    income = select_variable(background_vars, INCOME)

    income_labels = [
        "none",
        "under 15k",
        "15k-50k",
        "over 50k",
    ]

    return income.apply(
        lambda column: pd.cut(
            column,
            bins=[-np.inf, 0, 15000, 50000, np.inf],
            labels=income_labels,
            right=True,  # Relevant especially for 0 level
        )
    )


# Employment
# Derived from primary occupation
def merge_and_map_categories(column: pd.Series) -> pd.Series:
    # It would be nice to have fewer levels here for sparsity and stuff,
    # but I don't think it gets any less than this.
    EMPLOYED = "employed"
    UNEMPLOYED = "unemployed"
    HOMEMAKER = "homemaker"
    STUDENT = "student"
    RETIRED = "retired"
    UNABLE = "unable to work"

    # Map to new codes
    old_category_to_new_category = {
        "paid employment": EMPLOYED,
        "works or assists in family business": EMPLOYED,
        "autonomous professional, freelancer, or self-employed": EMPLOYED,
        "job seeker following job loss": UNEMPLOYED,
        "first-time job seeker": STUDENT,
        "exempted from job seeking following job loss": UNEMPLOYED,  # Very rare anyways, close enough to UNEMPLOYED
        "attends school or is studying": STUDENT,
        "takes care of the housekeeping": HOMEMAKER,
        "is pensioner ([voluntary] early retirement, old age pension scheme)": RETIRED,
        "has (partial) work disability": UNABLE,
        "performs unpaid work while retaining unemployment benefit": EMPLOYED,
        "performs voluntary work": EMPLOYED,
        "does something else": EMPLOYED,
        "is too young to have an occupation": STUDENT,
    }

    result = pd.Categorical(column.map(old_category_to_new_category))

    # Apparently if you set name=... here, it gets ignored because of .apply
    # Restore old index, otherwise things go NaN
    return pd.Series(result, index=column.index)


def make_employment(background_vars: pd.DataFrame) -> pd.DataFrame:
    occupation = select_variable(background_vars, PRINCIPAL_OCCUPATION)

    EMPLOYMENT = "employment"

    # Apply column-wise to have cohesive datatype
    employment = occupation.apply(merge_and_map_categories)

    columns: list[Column] = employment.columns  # pyright: ignore[reportAssignmentType]
    employment.columns = [Column(EMPLOYMENT, column.wave) for column in columns]

    return employment


# BMI
def make_bmi(health_panel: pd.DataFrame) -> pd.DataFrame:
    weight = select_variable(health_panel, WEIGHT)
    height = select_variable(health_panel, HEIGHT)

    # Broad sanity check
    TALLEST_HEIGHT_EVER = 270  # According to google idk
    VERY_SHORT_BABY = 5
    height = height.mask(height > TALLEST_HEIGHT_EVER, pd.NA)
    height = height.mask(height < VERY_SHORT_BABY, pd.NA)

    HEAVIEST_PERSON_EVER = 635  # According to google
    VERY_LIGHT_BABY = 1
    weight = weight.mask(weight > HEAVIEST_PERSON_EVER, pd.NA)
    weight = weight.mask(weight < VERY_LIGHT_BABY, pd.NA)

    bmi = pd.DataFrame(index=health_panel.index)

    BMI = "bmi"

    for year in available_years(weight):  # Can choose either weight or height, if one is missing answer is NA anyways
        bmi[Column(BMI, year)] = weight[Column(WEIGHT, year)] / (height[Column(HEIGHT, year)] / 100) ** 2

    # BMI ranges from https://www.who.int/europe/news-room/fact-sheets/item/a-healthy-lifestyle---who-recommendations
    return bmi.apply(
        lambda column: pd.cut(
            column,
            bins=[-np.inf, 18.5, 25.0, 30, np.inf],
            labels=["underweight", "normal weight", "overweight", "obese"],
            right=False,
        )
    )


# Depression status
def make_depression(health_panel: pd.DataFrame) -> pd.DataFrame:
    depression = select_variable(health_panel, DEPRESSION_MEDICATION)

    return depression.map(lambda x: x == "yes", na_action="ignore").astype("boolean")


# Previous depression
def make_previous_depression(health_panel: pd.DataFrame) -> pd.DataFrame:
    depression = select_variable(health_panel, DEPRESSION_MEDICATION)

    years_depression = sorted(available_years(depression))
    names_depression = [Column(DEPRESSION_MEDICATION, year) for year in sorted(available_years(depression))]

    previous_depression = pd.DataFrame(index=health_panel.index)
    PREVIOUS_DEPRESSION = "prev_depr"

    for person in previous_depression.index:
        yearly_medication_status: pd.Series = depression.loc[person, names_depression].squeeze()

        yearly_medication_status = yearly_medication_status.map(lambda x: x == "yes", na_action="ignore")

        cumulative_medication_status = pd.NA
        for year in years_depression[1:]:
            previous_year_name = Column(DEPRESSION_MEDICATION, year - 1)

            is_previous_year_available = pd.isna(yearly_medication_status.get(previous_year_name))

            if not is_previous_year_available:
                if pd.isna(cumulative_medication_status):
                    cumulative_medication_status = yearly_medication_status[previous_year_name]  # pyright: ignore  # noqa: PGH003
                else:
                    cumulative_medication_status = (
                        cumulative_medication_status or yearly_medication_status[previous_year_name]  # pyright: ignore  # noqa: PGH003
                    )

            previous_depression.loc[person, Column(PREVIOUS_DEPRESSION, year)] = cumulative_medication_status

    # Fix dtype, pandas doesn't automatically recognise a combination of NA and bool is just nullable boolean
    return previous_depression.astype("boolean")


# Education level
def make_education(background_vars: pd.DataFrame) -> pd.DataFrame:
    education = select_variable(background_vars, EDUCATION_LEVEL)

    category_map = {
        "havo/vwo (higher secondary education/preparatory university education, us: senior high school)": "havo vwo",
        "hbo (higher vocational education, us: college)": "hbo",
        "mbo (intermediate vocational education, us: junior college)": "mbo",
        "primary school": "primary school",
        "vmbo (intermediate secondary education, us: junior high school)": "vmbo",
        "wo (university)": "wo",
    }

    return education.apply(lambda column: column.cat.rename_categories(category_map))


# Ethnicity
def make_ethnicity(background_vars: pd.DataFrame) -> pd.DataFrame:
    ethnicity = select_variable(background_vars, ETHNICITY)

    category_map = {
        "dutch background": "dutch",
        "first generation foreign, non-western background": "first nonw",
        "first generation foreign, western background": "first w",
        "second generation foreign, non-western background": "second nonw",
        "second generation foreign, western background": "second w",
    }

    return ethnicity.apply(lambda column: column.cat.rename_categories(category_map))


# Finding the first non-NA value of each variable within each column
def find_first_non_na(data: pd.DataFrame) -> pd.Series:
    """Makes a series that contains the first (temporally) non-NA value in the data for each individual.
    Should be called with only one variable in `data`."""
    assert_column_type_correct(data)

    # Empty df
    if data.shape[1] == 0:
        logger.debug("first non-NA value of empty df, returning empty Series")
        return pd.Series(index=data.index)

    # Variable is constant
    if data.shape[1] == 1:
        logger.debug("first non-NA value of df with one column, returning column")
        return data.iloc[:, 0]

    variables = list({column.name for column in data.columns})  # pyright: ignore[reportAttributeAccessIssue]
    assert len(variables) == 1, (
        "Should be called with just the data for one variable across time, not multiple different variables,"
        f" but got {variables}"
    )

    variable_sorted = data[sorted(data.columns, key=lambda column: column.wave)]  # pyright: ignore[reportAttributeAccessIssue]

    # Already asserted all columns are same variable, so same dtype
    name = Column(f"{variables[0]}_first")
    result = pd.Series(index=data.index, name=name, dtype=data.dtypes.iloc[0])

    # Have to loop because df.first_valid_index() not vectorised, returns index of first row containing any valid value
    for individual in variable_sorted.index:
        index = variable_sorted.loc[individual, :].first_valid_index()

        value = pd.NA if index is None else variable_sorted.loc[individual, index]

        result.loc[individual] = value

    return result


def add_first_non_na(variable: pd.DataFrame) -> pd.DataFrame:
    """Makes a series that contains the first (temporally) non-NA value in the data for each individual.
    Returns a new dataframe with this series added."""

    new_series = find_first_non_na(variable)

    return pd.concat([variable, new_series], axis=1)


# The big merge
def make_dummies(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([series_to_dummies(df[column]) for column in df], axis=1)


def series_to_dummies(series: pd.Series) -> pd.DataFrame:
    assert isinstance(series, pd.Series), f"Expected a pd.Series, got {type(series)}:\n{series}"

    result = pd.get_dummies(series, prefix_sep=".", dtype="boolean")

    # Fix column names
    old_column: Column = series.name  # type: ignore  # noqa: PGH003
    result.columns = [replace(old_column, dummy_level=cleanup_dummy(dummy_level)) for dummy_level in result]  # type: ignore # noqa: PGH003

    # If original was NA, the result will have False in each column.
    # Replace that False with NA
    where_original_na = series.isna()
    result.loc[where_original_na, :] = pd.NA

    return result


def make_all_data(*, cache: bool, respect_load_cache: bool = True) -> pd.DataFrame:
    path = Path("../data/all_data.pkl")

    if cache and path.exists():
        return pd.read_pickle(path)  # noqa: S301

    # Data loading
    background_vars = load_wide_panel_cached("avars", respect_cache=respect_load_cache)
    leisure_panel = load_wide_panel_cached("cs", respect_cache=respect_load_cache)
    health_panel = load_wide_panel_cached("ch", respect_cache=respect_load_cache)

    # All variables (potentially) used in modelling
    mhi5 = make_mhi5(health_panel)
    sports = make_sports(leisure_panel)
    age = make_age(background_vars)
    ethnicity = make_ethnicity(background_vars)
    gender = select_variable(background_vars, GENDER)
    marital_status = select_variable(background_vars, MARITAL_STATUS)
    income = make_income(background_vars)
    education = make_education(background_vars)
    employment = make_employment(background_vars)
    physical_health = select_variable(health_panel, PHYSICAL_HEALTH)
    bmi = make_bmi(health_panel)
    depression = make_depression(health_panel)
    previous_depression = make_previous_depression(health_panel)

    # Add "*_first" version of each variable
    # Can't find a nice DRY way to do this, whatever
    mhi5 = add_first_non_na(mhi5)
    sports = add_first_non_na(sports)
    age = add_first_non_na(age)
    ethnicity = add_first_non_na(ethnicity)
    gender = add_first_non_na(gender)
    marital_status = add_first_non_na(marital_status)
    income = add_first_non_na(income)
    education = add_first_non_na(education)
    employment = add_first_non_na(employment)
    physical_health = add_first_non_na(physical_health)
    bmi = add_first_non_na(bmi)
    depression = add_first_non_na(depression)
    previous_depression = add_first_non_na(previous_depression)

    result = pd.DataFrame(index=background_vars.index).join(
        [
            mhi5,
            sports,
            make_dummies(age),
            make_dummies(ethnicity),
            make_dummies(gender),
            make_dummies(marital_status),
            make_dummies(income),
            make_dummies(education),
            make_dummies(employment),
            make_dummies(physical_health),
            make_dummies(bmi),
            depression,
            previous_depression,
        ]
    )

    result.to_pickle(path)

    return result
