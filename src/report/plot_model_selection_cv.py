# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def plot(  # noqa: PLR0913
    estimates: list[float],
    stds: list[float],
    sigma_estimates: list[float] | None = None,
    sigma_stds: list[float] | None = None,
    title: str = "Forecasting error for varying maximum AR lag",
    xlabel: str = "Number of AR lags",
):
    plt.figure()
    plt.title(title)
    plt.ylabel("RMSPE (MHI5)")
    plt.xlabel(xlabel)

    x = np.arange(len(estimates)) + 1

    plt.errorbar(x, estimates, yerr=stds, capsize=4)

    index_min = np.argmin(estimates)
    plt.axhline(estimates[index_min] + stds[index_min], color="red", alpha=0.5)

    if sigma_estimates is None or sigma_stds is None:
        plt.show()
        return

    total_uncertainty = np.sqrt(np.array(sigma_estimates) ** 2 + np.array(sigma_stds) ** 2)

    plt.errorbar(
        x,
        np.array(estimates) + np.array(stds),
        yerr=total_uncertainty,
        linestyle="None",
        color="red",
        alpha=0.5,
        capsize=2,
    )

    plt.show()


# %%
# AR lags 1-8 with distributed lags (DL) 1, R = 100 repeats, FIML (not fiml.x) and MLR
estimates = [10.55, 9.67, 9.42, 9.25, 9.19, 9.19, 9.05, 8.93]
stds = [0.36, 0.39, 0.37, 0.47, 0.43, 0.46, 0.51, 0.49]
sigma_estimates = [0.001, 0.001, 0.002, 0.012, 0.002, 0.002, 0.024, 0.003]
sigma_stds = [0.012, 0.015, 0.014, 0.029, 0.018, 0.017, 0.050, 0.017]

plot(estimates, stds, sigma_estimates, sigma_stds)

# %% Comparing 3 to 4 max lags on a more complex model
# Which uh, is kinda meaningless. 1-sigma rule has very different meaning here.
# The difference is comparable to the difference with only mhi5_23 though, so corroborates the choice in that regard.
# (IIRC) AR lags 3 and 4 with DL 1, R = 10 repeats, no FIML, no MLR

estimates_complex = [10.73, 10.53]
stds_complex = [0.37, 0.37]

plot(estimates_complex, stds_complex)

# %% Comparing X lags (distributed lags)
# Minimum lag 0
# (IIRC) DL 1-11 with AR 4, R = 10 repeats, no FIML or MLR

estimates_x = [9.25, 9.29, 9.29, 9.32, 9.26, 9.24, 9.09, 9.06, 9.03, 8.84, 8.85]
stds_x = [0.42, 0.46, 0.45, 0.45, 0.35, 0.44, 0.59, 0.49, 0.40, 0.57, 0.43]

plot(estimates, stds, title="Forecasting error for varying distributed lag", xlabel="Number of X lags")
