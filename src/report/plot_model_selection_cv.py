# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def plot(  # noqa: PLR0913
    estimates: list[float],
    stds: list[float],
    sigma_estimates: list[float] | None = None,
    sigma_stds: list[float] | None = None,
    title: str = "Forecasting error for varying AR lag order",
    xlabel: str = "Maximum $y$ lag",
    save_path: str | None = None,
):
    plt.figure(figsize=(4, 3))
    plt.title(title)
    plt.ylabel("RMSPE (MHI5)")
    plt.xlabel(xlabel)

    x = np.arange(len(estimates)) + 1

    plt.errorbar(x, estimates, yerr=stds, capsize=4, color="black")

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

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


# %%
# AR lags 1-8 with distributed lags (DL) [0, 1], R = 50 repeats, FIML (not fiml.x) and MLR
estimates = [10.16, 9.38, 9.18, 9.06, 9.01, 9.04, 8.83, 8.80]
stds = [0.35, 0.39, 0.38, 0.43, 0.43, 0.47, 0.44, 0.49]
sigma_estimates = [0.002, 0.003, 0.002, 0.002, 0.003, 0.032, 0.004, 0.026]
sigma_stds = [0.016, 0.015, 0.019, 0.024, 0.024, 0.059, 0.021, 0.066]

plot(estimates, stds, sigma_estimates, sigma_stds, save_path="../report/thesis/figures/modelling/cv_AR.svg")

# %% Comparing 3 to 4 max lags on a more complex model
# Which uh, is kinda meaningless. 1-sigma rule has very different meaning here.
# The difference is comparable to the difference with only mhi5_23 though, so corroborates the choice in that regard.
# (IIRC) AR lags 3 and 4 with DL 1, R = 10 repeats, no FIML, no MLR

estimates_complex = [10.73, 10.53]
stds_complex = [0.37, 0.37]

plot(estimates_complex, stds_complex)

# %% Comparing X lags (distributed lags)
# Minimum lag 0
# DL 1-11 with AR 3, R = 50 repeats, FIML (not fiml.x) and MLR

estimates_x = [9.18, 9.21, 9.19, 9.27, 9.23, 9.17, 9.02, 9.04, 8.95, 8.79, 8.80]
stds_x = [0.36, 0.39, 0.38, 0.54, 0.56, 0.42, 0.44, 0.53, 0.48, 0.54, 0.52]
sigma_estimates_x = [0.003, 0.003, 0.003, 0.038, 0.040, 0.003, 0.004, 0.037, 0.029, 0.028, 0.026]
sigma_stds_x = [0.016, 0.017, 0.019, 0.084, 0.083, 0.024, 0.022, 0.082, 0.058, 0.065, 0.056]

plot(
    estimates_x,
    stds_x,
    sigma_estimates_x,
    sigma_stds_x,
    title="Forecasting error for varying DL lag order",
    xlabel="Maximum $x$ lag",
    save_path="../report/thesis/figures/modelling/cv_DL.svg",
)
