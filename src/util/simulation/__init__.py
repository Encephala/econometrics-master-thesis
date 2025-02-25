import numpy as np

rng = np.random.default_rng()


def take_lagged_subset(values: np.ndarray, current_t: int, num_lags: int) -> np.ndarray:
    """For correlating one variable to another.
    Takes a subset of passed values, such that the values at `current_t` and the `num_lags` values before it are returned.
    Does not reorder, i.e. maintains chronological order in the resulting array.
    """
    # Lil sanity check
    assert len(values.shape) == 2

    print(f"{current_t=}, {num_lags=}")

    return values[:, max(0, current_t - num_lags) : current_t + 1]


class DataGenerator:
    AR_lags: list[float]
    lags_x_for_y: list[float]
    lags_y_for_x: list[float]

    def __init__(self, AR_lags: list[float], lags_x_for_y: list[float], lags_y_for_x: list[float]):
        """Pass the lag coefficients, implicitly defining the lag structure by the nonzero coefficients.

        I.e. not [1, 0, 1] but [0.5, 0, 0.8]."""
        self.AR_lags = AR_lags
        self.lags_x_for_y = lags_x_for_y
        self.lags_y_for_x = lags_y_for_x

        if len(lags_x_for_y) != 0 and lags_x_for_y[0] != 0 and len(lags_y_for_x) != 0 and lags_y_for_x[0] != 0:
            # Because the order in which x influences y or y influences x is hardcoded in self.generate,
            # that might not be what is expected by future me.
            # But not hard-coding is hard so screw this edge case
            raise ValueError("Undefined behaviour if x and y both instantaneously influence each other")

    def generate(
        self,
        size: tuple[int, int],
        *,
        # Prime numbers so I can recognise them/their products in outputs
        sigma_y: float = np.sqrt(3),
        sigma_x: float = np.sqrt(5),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (x, y)."""
        # Convenience aliases
        N, T = size

        y_for_x = self.lags_y_for_x
        x_for_y = self.lags_x_for_y
        AR_lags = [0] + self.AR_lags

        # Draw baseline values for y and x
        # Baseline as in without the mutual influences
        y = rng.normal(0, sigma_y, size)
        x = rng.normal(0, sigma_x, size)

        # Apply the mutual influence
        # This formula works for both the initial values (when fewer than max_lag historic values are available)
        # and the general case (when there is sufficient history for the full lag structure)
        for t in range(T):
            print(f"Generating for {t=}")

            # AR

            # TODO
            # y[:, : t + 1] += np.array(AR_lags) * roll_and_take_subset(
            #     y, min(t, max_AR_lag), max(0, t - )
            # )

            # Mutual influence

            lagged_x_subset = take_lagged_subset(
                x,
                t,
                min(t, len(x_for_y) - 1),  # -1 because the zeroth lag is included, so furthest lag is length - 1
            )
            # This broadcasts the coefficients across the first axis (the cohort/the variable i),
            # so that at each timepoint each observation gets the right coefficient
            y[:, t] += np.sum(np.flip(x_for_y[: t + 1]) * lagged_x_subset, axis=1)

            # Same as logic for x influencing y
            lagged_y_subset = take_lagged_subset(
                y,
                t,
                min(t, len(y_for_x) - 1),
            )
            x[:, t] += np.sum(
                np.flip(y_for_x[: t + 1]) * lagged_y_subset,
                axis=1,
            )

        return x, y
