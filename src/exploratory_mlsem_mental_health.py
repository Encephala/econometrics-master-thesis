#!/usr/bin/env python3

# %% imports
import semopy
import pandas as pd

from util import ModelDefinitionBuilder, load_wide_panel_cached, standardise_wide_column_name, select_question_wide

# %% consts
HAPPINESS = "ch15"
FITNESS = "cs121"

# %% data loading
leisure_panel = load_wide_panel_cached("cs").rename(columns=standardise_wide_column_name)
health_panel = load_wide_panel_cached("ch").rename(columns=standardise_wide_column_name)


# %% selecting columns
# Actually have to do this because the data sometimes (but not consistently, smile) has prefixed spaces,
# and sometimes has capitalisation
def fix_column_categories(column: pd.Series) -> pd.Series:
    old_categories: "pd.Index[str]" = column.cat.categories  # To help the LSP

    new_categories = pd.Index([category.lower().strip() for category in old_categories])

    return column.cat.rename_categories(new_categories)


happiness = select_question_wide(health_panel, HAPPINESS)
happiness = happiness.apply(
    lambda column: pd.Categorical(
        fix_column_categories(column),
        categories=["never", "seldom", "sometimes", "often", "mostly", "continuously"],
        ordered=True,
    ),  # pyright: ignore[reportCallIssue, reportArgumentType]
)

fitness = select_question_wide(leisure_panel, FITNESS)
fitness = fitness.apply(
    lambda column: pd.Categorical(fix_column_categories(column), categories=["no", "yes"], ordered=True)  # pyright: ignore[reportCallIssue, reportArgumentType]
)

# %% build simple semopy model
complete_data = pd.concat([happiness, fitness], join="outer", axis="columns").apply(
    lambda column: pd.Series(column).cat.codes.map(lambda value: float("nan") if value == -1 else value)
)
model_definition = (
    ModelDefinitionBuilder()
    .with_y(HAPPINESS, ordinal=True, lag_structure=[1, 2])
    .with_x(FITNESS, lag_structure=[0, 1, 2])
    .build(complete_data.columns)
)

print(model_definition)

model = semopy.Model(model_definition)

# %% fit the model
result = model.fit(complete_data, obj="FIML")
