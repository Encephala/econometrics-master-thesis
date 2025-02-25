#!/usr/bin/env python3
# %%
from util import simulation

# %%
N = 500
T = 7

generator = simulation.DataGenerator([0.5, 0.3], [1, 0.9, 0.8, 0.7], [0, 0.1, 0.2])

# %%
data = generator.generate((N, T))
