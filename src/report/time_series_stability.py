# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

N = 1000


# %%
# `coefs` from most recent lag to least recent
def empirical(coefs):
    data = np.empty(N)

    for i in range(len(coefs)):
        data[i] = 1

    for i in range(len(coefs), N):
        data[i] = np.sum(data[i - len(coefs) : i] * np.flip(coefs))

    plt.figure()
    plt.plot(data)
    plt.title(str(coefs))
    plt.show()


def analytical(coefs):
    roots = np.roots([*np.flip(coefs), -1])

    print(roots)

    plt.figure()
    plt.scatter(roots.real, roots.imag)

    plt.gca().add_patch(Circle((0, 0), 1, fill=False))

    plt.show()


# %%
analytical([0.396, 0.247, 0.202])
