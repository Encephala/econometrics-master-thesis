from typing import Self, Tuple
from dataclasses import dataclass, field
import warnings

import pandas as pd


@dataclass(frozen=True)
class Variable:
    name: str
    named_parameter: str | None = field(default=None, compare=False)

    def __str__(self) -> str:
        if self.named_parameter is not None:
            return f"{self.named_parameter}*{self.name}"

        return f"{self.name}"


@dataclass(frozen=True)
class Regression:
    lval: Variable
    rvals: list[Variable]

    def __str__(self) -> str:
        return f"{self.lval} ~ {' + '.join(map(str, self.rvals))}"

    def __hash__(self) -> int:
        return self.lval.__hash__() + sum((i + 1) * val.__hash__() for i, val in enumerate(self.rvals))


@dataclass(frozen=True)
class Measurement:
    lval: Variable
    rvals: list[Variable]

    def __str__(self) -> str:
        return f"{self.lval} =~ {' + '.join(map(str, self.rvals))}"

    def __hash__(self) -> int:
        return self.lval.__hash__() + sum((i + 1) * val.__hash__() for i, val in enumerate(self.rvals))


@dataclass(frozen=True)
class Covariance:
    lval: Variable
    rvals: list[Variable]

    def __str__(self) -> str:
        return f"{self.lval} ~~ {' + '.join(map(str, self.rvals))}"

    def __hash__(self) -> int:
        return self.lval.__hash__() + sum((i + 1) * val.__hash__() for i, val in enumerate(self.rvals))


class OrdinalVariableSet(set[Variable]):
    def build(self) -> str:
        if len(self) == 0:
            return ""

        return f"DEFINE(ordinal) {' '.join(sorted(variable.name for variable in self))}"


class ModelDefinitionBuilder:
    y_name: str
    y_ordinal: bool
    y_lag_structure: list[int]

    x_name: str
    x_ordinal: bool
    x_lag_structure: list[int]

    w_names: list[str]
    w_ordinal: list[bool]

    def __init__(self):
        self._regressions: set[Regression] = set()
        self._measurements: set[Measurement] = set()
        self._covariances: set[Covariance] = set()
        self._ordinals = OrdinalVariableSet()

        self.w_names = []

    def with_y(self, name: str, *, lag_structure: list[int] | None = None, ordinal: bool = False) -> Self:
        self.y_name = name
        self.y_ordinal = ordinal
        self.y_lag_structure = lag_structure if lag_structure is not None else [1]
        return self

    def with_x(self, name: str, *, lag_structure: list[int] | None = None, ordinal: bool = False) -> Self:
        self.x_name = name
        self.x_ordinal = ordinal
        self.x_lag_structure = lag_structure if lag_structure is not None else [1]
        return self

    def with_w(self, names: list[str], ordinal: list[bool] | None = None) -> Self:
        self.w_names = names
        self.w_ordinal = ordinal if ordinal is not None else [False] * len(names)
        return self

    def build(self, available_columns: "pd.Index[str]") -> str:
        first_year_y, last_year_y, _, last_year_x = self._determine_start_and_end_years(available_columns)

        self._build_regressions(available_columns, first_year_y, last_year_y)

        self._make_x_predetermined(first_year_y, last_year_x)

        return f"""# Regressions (structural part)
{"\n".join([*sorted(map(str, self._regressions)), ""])}
# Measurement part
{"\n".join([*sorted(map(str, self._measurements)), ""])}
# Additional covariances
{"\n".join([*sorted(map(str, self._covariances)), ""])}
# Operations/constraints
{self._ordinals.build()}
"""

    def _determine_start_and_end_years(self, available_columns: "pd.Index[str]") -> Tuple[int, int, int, int]:
        y_years = [
            int(column[column.find("_") + 1 :]) for column in available_columns if column.startswith(self.y_name)
        ]
        y_start = min(y_years)
        y_end = max(y_years)

        x_years = [
            int(column[column.find("_") + 1 :]) for column in available_columns if column.startswith(self.x_name)
        ]
        x_start = min(x_years)
        x_end = max(x_years)

        # TODO?: When using FIML/handling missing data, perhaps the `max` and `min` in these two statements should swap,
        # but then we have to also add those columns to the df to prevent index errors.
        # If x goes far enough back, the first regression is when y starts
        # else, start as soon as we can due to x
        first_year_y = max(y_start + max(self.y_lag_structure), x_start + max(self.x_lag_structure))
        first_year_x = first_year_y - max(self.x_lag_structure)
        # If x does not go far enough forward, stop when x_stops
        # else, stop when y stops
        last_year_y = min(x_end + min(self.x_lag_structure), y_end)
        last_year_x = last_year_y - min(self.x_lag_structure)

        return first_year_y, last_year_y, first_year_x, last_year_x

    def _build_regressions(self, available_columns: "pd.Index[str]", first_year_y: int, last_year_y: int):
        for year_y in range(first_year_y, last_year_y + 1):
            y = Variable(f"{self.y_name}_{year_y}")

            y_lags = [Variable(f"{self.y_name}_{year_y - lag}", f"rho{lag}") for lag in self.y_lag_structure]
            x_lags = [Variable(f"{self.x_name}_{year_y - lag}", f"beta{lag}") for lag in self.x_lag_structure]

            w = [Variable(f"{name}_{year_y}", f"delta0_{j}") for j, name in enumerate(self.w_names)]

            if (
                any(variable.name not in available_columns for variable in y_lags)
                or any(variable.name not in available_columns for variable in x_lags)
                or any(variable.name not in available_columns for variable in w)
            ):
                warnings.warn(
                    f"For {year_y=}, one of {y_lags=}, {x_lags=} or {w=} was not in the data",
                    stacklevel=2,
                )
                # TODO: Is there a better option than ditching the whole regression because one of the vars is missing?
                continue

            rvals = [*y_lags, *x_lags, *w]
            self._regressions.add(Regression(y, rvals))

            # All y_lags and x_lags are included as regressors,
            # so don't have to check with self._regressions_contain here
            if self.y_ordinal:
                self._ordinals.update(y_lags)

            if self.x_ordinal:
                self._ordinals.update(x_lags)

            for variable, is_ordinal in zip(w, self.w_ordinal, strict=True):
                if is_ordinal:
                    self._ordinals.add(variable)

            # Fix variance for y to be constant in time
            self._covariances.add(Covariance(y, [Variable(y.name, "sigma")]))

    # Allow for pre-determined variables, i.e. arbitrary correlation between x and previous values of y
    # NOTE: The very first year of y in the data is considered exogenous, but first_year_y is the first regression
    # that we do, so this range correctly starts for the first endogenous y
    def _make_x_predetermined(self, first_year_y: int, last_year_x: int):
        for year in range(first_year_y, last_year_x):  # TODO: This doesn't properly respect variable lag structure
            y_current = Variable(f"{self.y_name}_{year}")
            x_future = [
                Variable(f"{self.x_name}_{future_year}", f"gamma{i}")
                for i, future_year in enumerate(range(year + 1, last_year_x + 1))
            ]

            # Sanity check that no y_years are considered that shouldn't be
            # (doesn't check for y_years that aren't included but should be)
            assert len(x_future) != 0, "Error in determining covariances to make x predetermined"

            # Filter future x's that don't exist in the regressions,
            # for instance due to data missing for a year
            # TODO: Rather than having this check, the loop should probably just run over the regressions.
            # But then how do I maintan temporal order of regressions?
            # Why do I even use a set xdd
            x_future = [variable for variable in x_future if self._regressions_contain(variable)]

            if len(x_future) != 0:
                self._covariances.add(Covariance(y_current, x_future))

    def _regressions_contain(self, variable: Variable) -> bool:
        return any(variable in regression.rvals for regression in self._regressions)
