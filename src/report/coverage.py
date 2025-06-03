# %%
from lib.data import make_all_data, available_waves, Column

# ruff: noqa: F403, F405
from lib.data.variables import *

data = make_all_data(cache=True)

# %%
for wave in available_waves(data):
    print(f"{wave=}")

    try:
        mhi5 = data[Column(MHI5, wave)]
    except:  # noqa: E722
        print("Not found mhi5")
        continue

    try:
        sports = data[Column(SPORTS, wave)]
    except:  # noqa: E722
        print("Not found sports")
        continue

    coverage = (~mhi5.isna()) & (~sports.isna())

    print(f"{coverage.mean():.2%}")
