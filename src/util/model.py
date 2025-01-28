from typing import Self, Tuple
import warnings

import pandas as pd


class ModelDefinitionBuilder:
    y: str
    y_ordinal: bool

    x: str
    x_ordinal: bool

    w: list[str]  # TODO
    lag_structure: list[int]

    def __init__(self):
        self._regressions: list[str] = []
        self._measurements: list[str] = []
        self._covariances: list[str] = []
        self._operations: list[str] = []

        self.w = []
        self.lag_structure = [1]

    # TODO?: implement some stuff to make it less likely that I forget to properly order an ordinal column,
    # since imported categorical data has arbitrary category to index coding
    def with_y(self, y: str, *, ordinal: bool = False) -> Self:
        self.y = y
        self.y_ordinal = ordinal
        return self

    def with_x(self, x: str, *, ordinal: bool = False) -> Self:
        self.x = x
        self.x_ordinal = ordinal
        return self

    def with_w(self, w: list[str]) -> Self:
        raise NotImplementedError

        self.w = w
        return self

    def with_lag_structure(self, lag_structure: list[int]) -> Self:
        self.lag_structure = lag_structure
        return self

    def build(self, available_columns: "pd.Index[str]") -> str:
        first_year_y, last_year_y, _, last_year_x = self._determine_start_and_end_years(available_columns)

        self._make_x_predetermined(first_year_y, last_year_x)

        self._build_regressions(available_columns, first_year_y, last_year_y)

        return f"""# Regressions (structural part)
{"\n".join([*self._regressions, ""])}
# Measurement part
{"\n".join([*self._measurements, ""])}
# Additional covariances
{"\n".join([*self._covariances, ""])}
# Operations/constraints
{"\n".join(self._operations)}
"""

    def _determine_start_and_end_years(self, available_columns: "pd.Index[str]") -> Tuple[int, int, int, int]:
        y_years = [int(column[column.find("_") + 1 :]) for column in available_columns if column.startswith(self.y)]
        y_start = min(y_years)
        y_end = max(y_years)

        x_years = [int(column[column.find("_") + 1 :]) for column in available_columns if column.startswith(self.x)]
        x_start = min(x_years)
        x_end = max(x_years)

        # TODO?: When using FIML/handling missing data, perhaps the `max` and `min` in these two statements should swap,
        # but then we have to also add those columns to the df to prevent index errors
        # If x goes far enough back, the first regression is when y starts
        # else, start as soon as we can due to x
        # TODO: Update this so that it is correct when we include a lag structure for y
        first_year_y = max(y_start, x_start + max(self.lag_structure))
        first_year_x = first_year_y - max(self.lag_structure)
        # If x does not go far enough forward, stop when x_stops
        # else, stop when y stops
        last_year_y = min(x_end + min(self.lag_structure), y_end)
        last_year_x = last_year_y - min(self.lag_structure)

        return first_year_y, last_year_y, first_year_x, last_year_x

    # Allow for pre-determined variables, i.e. arbitrary correlation between x and previous values of y
    # NOTE: The very first year of y in the data is considered exogenous, but first_year_y is the first regression
    # that we do, so this range correctly starts for the first endogenous y
    def _make_x_predetermined(self, first_year_y: int, last_year_x: int):
        for year in range(first_year_y, last_year_x):  # TODO: This doesn't properly respect variable lag structure
            y_current_name = f"{self.y}_{year}"
            x_future_names = [
                f"gamma{i}*{self.x}_{future_year}" for i, future_year in enumerate(range(year + 1, last_year_x + 1))
            ]

            # Sanity check that no y_years are included that shouldn't be
            # (doesn't check for y_years that aren't included but should be)
            assert len(x_future_names) != 0, "Error in determining covariances to make x predetermined"

            covariance = f"{y_current_name} ~~ {' + '.join(x_future_names)}"
            self._covariances.append(covariance)

    def _build_regressions(self, available_columns: "pd.Index[str]", first_year_y: int, last_year_y: int):
        for year_y in range(first_year_y, last_year_y + 1):
            y_name = f"{self.y}_{year_y}"
            x_names = [f"{self.x}_{year_y - lag}" for lag in self.lag_structure]

            if y_name not in available_columns or any(x_name not in available_columns for x_name in x_names):
                warnings.warn(f"For {year_y=}, either {y_name} or one of {x_names} was not in the data", stacklevel=2)
                continue

            rvals = [
                f"rho0*{self.y}_{year_y - 1}",
                *(f"beta{i}*{name}" for i, name in zip(self.lag_structure, x_names, strict=True)),
            ]

            regression = f"{y_name} ~ {' + '.join(rvals)}"
            self._regressions.append(regression)

            if self.y_ordinal:
                self._operations.append(f"DEFINE(ordinal) {y_name}")

            if self.x_ordinal:
                self._operations.append(f"DEFINE(ordinal) {' '.join(x_names)}")

            # Fix variance for y to be constant in time
            self._covariances.append(f"{y_name} ~~ sigma*{y_name}")
