#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt

from util.spss_load import load_df

health = load_df(Path("ch23p_EN_1.0p.csv"))
handedness = load_df(Path("db10a_EN_1.0p.sav"))

merged = health[["nomem_encr", "ch23p015"]].merge(handedness[["nomem_encr", "db10a003"]])

print(merged)

plot = merged.plot()
plt.show()
