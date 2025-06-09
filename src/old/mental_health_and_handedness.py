#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

from lib.data import load_df

LEFT_HANDED = 1.0
RIGHT_HANDED = 2.0

health = load_df(Path("ch23p_EN_LEFT_HANDEDp.sav"))
handedness = load_df(Path("db10a_EN_LEFT_HANDEDp.sav"))

merged = (
    health[["nomem_encr", "ch23p015"]]
    .merge(handedness[["nomem_encr", "db10a003"]])
    .rename({"ch23p015": "Mental Health", "db10a003": "Handedness"}, axis=1)
)
merged = merged[merged["Mental Health"] != " "].astype(
    {
        "Mental Health": "int64",
        "Handedness": "int64",
    }
)

print(merged)

print(merged[merged["Handedness"] == LEFT_HANDED]["Mental Health"].describe())
print(
    (
        merged[merged["Handedness"] == LEFT_HANDED]["Mental Health"].value_counts()
        / (merged["Handedness"] == LEFT_HANDED).sum()
    ).sort_index()
)

print(merged[merged["Handedness"] == RIGHT_HANDED]["Mental Health"].describe())
print(
    (
        merged[merged["Handedness"] == RIGHT_HANDED]["Mental Health"].value_counts()
        / (merged["Handedness"] == RIGHT_HANDED).sum()
    ).sort_index()
)

left_hand_stats = merged[merged["Handedness"] == LEFT_HANDED]["Mental Health"].describe()
right_hand_stats = merged[merged["Handedness"] == RIGHT_HANDED]["Mental Health"].describe()

print(left_hand_stats)

diff = right_hand_stats["mean"] - left_hand_stats["mean"]

left_std = left_hand_stats["std"] / np.sqrt(left_hand_stats["count"])
right_std = right_hand_stats["std"] / np.sqrt(right_hand_stats["count"])

# Approximately
std = (left_hand_stats["count"] * left_std + right_hand_stats["count"] * right_std) / (
    left_hand_stats["count"] + right_hand_stats["count"]
)

print(f"t-value: {diff / std} ({diff=}, {std=})")
print(f"p = {2 * sps.norm.sf(diff / std)}")

# plt.figure(1)
# plt.title("Left-handed")
# plt.hist(merged[merged["Handedness"] == LEFT_HANDED]["Mental Health"], bins = np.arange(0.5, 7.5)) # type: ignore

# plt.figure(2)
# plt.title("Right-handed")
# plt.hist(merged[merged["Handedness"] == RIGHT_HANDED]["Mental Health"], bins = np.arange(0.5, 7.5)) # type: ignore

# plt.show()
