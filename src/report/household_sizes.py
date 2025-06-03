# %%
from lib.data import load_wide_panel_cached, select_variable

health = load_wide_panel_cached("ch")

# %%
household = select_variable(health, "nohouse_encr")

print((len(household) - household.isna().sum()) / household.nunique(dropna=True))
