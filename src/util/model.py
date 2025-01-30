from typing import Self, Tuple
from dataclasses import dataclass, field
import warnings

import pandas as pd


@dataclass(frozen=True)
class Variable:
    name: str
    wave: int
    named_parameter: str | None = field(default=None, compare=False)

    def full_name(self) -> str:
        return f"{self.name}_{self.wave}"

    def __str__(self) -> str:
        if self.named_parameter is not None:
            return f"{self.named_parameter}*{self.full_name()}"

        return self.full_name()


@dataclass(frozen=True)
class Regression:
    lval: Variable
    rvals: list[Variable]
    wave: int

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

        return f"DEFINE(ordinal) {' '.join(sorted(variable.full_name() for variable in self))}"


class ModelDefinitionBuilder:
    y_name: str
    y_ordinal: bool
    y_lag_structure: list[int]

    x_name: str
    x_ordinal: bool
    x_lag_structure: list[int]

    w_names: list[str] | None = None
    w_ordinal: list[bool] | None = None

    def __init__(self):
        self._regressions: list[Regression] = []
        self._measurements: list[Measurement] = []
        self._covariances: list[Covariance] = []
        self._ordinals = OrdinalVariableSet()

        self.w_names = []

    def with_y(self, name: str, *, lag_structure: list[int] | None = None, ordinal: bool = False) -> Self:
        self.y_name = name
        self.y_ordinal = ordinal
        if lag_structure is not None:
            # Zero lag breaks the regression, negative lags break the logic for when the first/last regressions are
            assert all(i > 0 for i in lag_structure), "Invalid lags provided for y"
            self.y_lag_structure = lag_structure
        else:
            self.y_lag_structure = [1]
        return self

    def with_x(self, name: str, *, lag_structure: list[int] | None = None, ordinal: bool = False) -> Self:
        self.x_name = name
        self.x_ordinal = ordinal
        if lag_structure is not None:
            # Negative lags break the logic for when the first/last regressions are
            assert all(i >= 0 for i in lag_structure), "Invalid lags provided for x"
            self.x_lag_structure = lag_structure
        else:
            self.x_lag_structure = [1]
        return self

    def with_w(self, names: list[str], ordinal: list[bool] | None = None) -> Self:
        self.w_names = names
        self.w_ordinal = ordinal if ordinal is not None else [False] * len(names)
        return self

    def build(self, available_columns: "pd.Index[str]") -> str:
        available_variables = [
            Variable(column[: column.rfind("_")], int(column[column.rfind("_") + 1 :])) for column in available_columns
        ]

        first_year_y, last_year_y = self._determine_start_and_end_years(available_variables)

        self._build_regressions(available_variables, first_year_y, last_year_y)

        self._make_x_predetermined()

        return f"""# Regressions (structural part)
{"\n".join([*map(str, self._regressions), ""])}
# Measurement part
{"\n".join([*map(str, self._measurements), ""])}
# Additional covariances
{"\n".join([*map(str, self._covariances), ""])}
# Operations/constraints
{self._ordinals.build()}
"""

    def _determine_start_and_end_years(self, available_variables: "list[Variable]") -> Tuple[int, int]:
        y_years = [variable.wave for variable in available_variables if variable.name == self.y_name]
        y_start = min(y_years)
        y_end = max(y_years)

        x_years = [variable.wave for variable in available_variables if variable.name == self.x_name]
        x_start = min(x_years)
        x_end = max(x_years)

        # TODO?: When using FIML/handling missing data, perhaps the `max` and `min` in these two statements should swap,
        # but then we have to also add those columns to the df to prevent index errors.
        # If x goes far enough back, the first regression is when y starts
        # else, start as soon as we can due to x
        first_year_y = max(y_start + max(self.y_lag_structure), x_start + max(self.x_lag_structure))
        # If x does not go far enough forward, stop when x_stops
        # else, stop when y stops
        last_year_y = min(x_end + min(self.x_lag_structure), y_end)

        return first_year_y, last_year_y

    def _build_regressions(self, available_variables: "list[Variable]", first_year_y: int, last_year_y: int):
        for year_y in range(first_year_y, last_year_y + 1):
            y = Variable(self.y_name, year_y)

            if y not in available_variables:
                warnings.warn(f"{y=} not found in data, skipping regression", stacklevel=2)
                continue

            y_lags = [Variable(self.y_name, year_y - lag, f"rho{lag}") for lag in self.y_lag_structure]
            x_lags = [Variable(self.x_name, year_y - lag, f"beta{lag}") for lag in self.x_lag_structure]

            w = (
                [Variable(name, year_y, f"delta0_{j}") for j, name in enumerate(self.w_names)]
                if self.w_names is not None
                else []
            )

            if (
                any(variable not in available_variables for variable in y_lags)
                or any(variable not in available_variables for variable in x_lags)
                or any(variable not in available_variables for variable in w)
            ):
                warnings.warn(
                    f"For {y=}, one of {y_lags=}, {x_lags=} or {w=} was not in the data, skipping regression",
                    stacklevel=2,
                )
                # TODO: Is there a better option than ditching the whole regression because one of the vars is missing?
                continue

            rvals = [*y_lags, *x_lags, *w]
            self._regressions.append(Regression(y, rvals, year_y))

            # All y_lags and x_lags are included as regressors,
            # so don't have to check with self._regressions_contain here
            if self.y_ordinal:
                self._ordinals.update(y_lags)

            if self.x_ordinal:
                self._ordinals.update(x_lags)

            w_ordinal = self.w_ordinal if self.w_ordinal is not None else []
            for variable, is_ordinal in zip(w, w_ordinal, strict=True):
                if is_ordinal:
                    self._ordinals.add(variable)

            # Fix variance for y to be constant in time
            self._covariances.append(Covariance(y, [Variable(y.name, y.wave, "sigma")]))

    # Allow for pre-determined variables, i.e. arbitrary correlation between x and previous values of y
    # NOTE: The very first value of y in the data is considered exogenous and thus it can't be correlated with future x
    def _make_x_predetermined(self):
        all_regressors = [rval for regression in self._regressions for rval in regression.rvals]

        for regression in self._regressions:
            y_current = regression.lval
            x_future = [
                Variable(variable.name, variable.wave, f"gamma{variable.wave - regression.wave}")
                for variable in all_regressors
                if variable.name == self.x_name and variable.wave > regression.wave
            ]

            if len(x_future) != 0:
                self._covariances.append(Covariance(y_current, x_future))
