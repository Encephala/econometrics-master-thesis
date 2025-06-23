# %% imports
import pandas as pd

from lib.data import make_all_data, load_wide_panel_cached, select_variable, available_waves, select_wave

# ruff: noqa: F403, F405
from lib.data.variables import *

# %%
health = load_wide_panel_cached("ch")
leisure = load_wide_panel_cached("cs")
background = load_wide_panel_cached("avars")

all_data = make_all_data(cache=True)


# %%
def print_correlations(left: pd.DataFrame, right: pd.DataFrame):
    for wave in available_waves(pd.concat([left, right], axis=1)):
        print(f"{wave=}")

        try:
            left_var = select_wave(left, wave)
        except KeyError:
            print("Left not available")
            continue

        try:
            right_var = select_wave(right, wave)
        except KeyError:
            print("Right not available")
            continue

        print(pd.concat([left_var, right_var], axis=1).corr())


# %% MHI5 and sports
mhi5 = select_variable(all_data, MHI5)
sports = select_variable(all_data, SPORTS)

print_correlations(mhi5, sports)

# %% Disease and medication
disease = select_variable(all_data, DISEASE_STATUS)
medication = select_variable(health, "ch184")
medication = medication.map(lambda x: x == "yes", na_action="ignore")

print_correlations(disease, medication)

# %% Disease and physical health
physical_health = select_variable(all_data, PHYSICAL_HEALTH)

print_correlations(disease, physical_health)
