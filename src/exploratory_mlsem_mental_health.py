#!/usr/bin/env python3

# %% imports
import semopy

from util import load_wide_panel_cached, standardise_wide_column, select_question_wide

# %% consts
HAPPINESS = "ch15"
FITNESS = "cs121"

# %% data loading
leisure_panel = load_wide_panel_cached("cs").rename(columns=standardise_wide_column)
health_panel = load_wide_panel_cached("ch").rename(columns=standardise_wide_column)

# %% selecting columns
happiness = select_question_wide(health_panel, HAPPINESS)
fitness = select_question_wide(leisure_panel, FITNESS)
