#!/usr/bin/env python3

#%%
from util import load_panel_cached

#%%
leisure_panel = load_panel_cached("cs")
health_panel = load_panel_cached("ch")

print(leisure_panel)
print(health_panel)
