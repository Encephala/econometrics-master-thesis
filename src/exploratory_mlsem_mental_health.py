#!/usr/bin/env python3

# %% imports
import semopy
import pandas as pd

from util import ModelBuilder, load_wide_panel_cached, standardise_wide_column, select_question_wide

# %% consts
HAPPINESS = "ch15"
FITNESS = "cs121"

# %% data loading
leisure_panel = load_wide_panel_cached("cs").rename(columns=standardise_wide_column)
health_panel = load_wide_panel_cached("ch").rename(columns=standardise_wide_column)

# %% selecting columns
happiness = select_question_wide(health_panel, HAPPINESS)
happiness = happiness.apply(
    lambda column: pd.Categorical(
        column, categories=["never", "seldom", "sometimes", "often", "mostly", "continuously"], ordered=True
    ),  # pyright: ignore[reportCallIssue, reportArgumentType]
)

fitness = select_question_wide(leisure_panel, FITNESS)
fitness = fitness.apply(lambda column: pd.Categorical(column, categories=["no", "yes"], ordered=True))  # pyright: ignore[reportCallIssue, reportArgumentType]

# %% build quick-and-dirty semopy model
# TODO: cast categorical data to ints and then tell model that it's ordinal stuff
complete_data = pd.concat([happiness, fitness], join="outer", axis="columns").apply(lambda column: column.cat.codes)
model_definition = ModelBuilder().with_y(HAPPINESS, ordinal=True).with_x(FITNESS, ordinal=True).build(complete_data)

model = semopy.Model(model_definition)

# %% fit the model
result = model.fit(complete_data)

# %%
