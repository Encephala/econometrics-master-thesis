#!/usr/bin/env python3

#%% imports
import semopy

from util import load_wide_panel_cached, standardise_wide_column

#%% data loading
leisure_panel = load_wide_panel_cached("cs").rename(columns = standardise_wide_column)
health_panel = load_wide_panel_cached("ch").rename(columns = standardise_wide_column)

#%%
# model = semopy.Model()

# %%
