from typing import Self, Sequence, Tuple
from dataclasses import dataclass, field
import warnings

import pandas as pd


@dataclass(frozen=True)
class VariableDefinition:
    "A variable in the model."

    name: str
    is_ordinal: bool = field(default=False, kw_only=True)
    is_dummy: bool = field(default=False, kw_only=True)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class VariableInWave:
    "A variable in the dataset, that is, a specific variable in a specific wave."

    variable: VariableDefinition
    wave: int

    def __str__(self):
        return f"{self.variable}_{self.wave}"


@dataclass(frozen=True)
class VariableWithNamedParameter(VariableInWave):
    "A variable in the dataset that has an associated named parameter."

    parameter: str

    def __str__(self) -> str:
        return f"{self.parameter}*{super().__str__()}"


@dataclass(frozen=True)
class AvailableVariable:
    "A variable as available in the data, effectively a parsed column name."

    name: str
    wave: int
    dummy_level: str | None = field(default=None)

    @classmethod
    def from_column_name(cls, column: str) -> Self:
        name = column[: column.rfind("_")]

        dummy_level = column[: column.rfind("|") + 1 :] if name.find("|") != -1 else None

        if dummy_level is None:
            wave = int(column[column.rfind("_") + 1 :])
        else:
            wave = int(column[column.rfind("_") + 1 : column.find("|")])

        return cls(name, wave, dummy_level)


@dataclass(frozen=True)
class Regression:
    lval: VariableInWave
    rvals: Sequence[VariableInWave]
    wave: int

    def __str__(self) -> str:
        return f"{self.lval} ~ {' + '.join(map(str, self.rvals))}"

    def __hash__(self) -> int:
        return self.lval.__hash__() + sum((i + 1) * val.__hash__() for i, val in enumerate(self.rvals))


@dataclass(frozen=True)
class Measurement:
    lval: VariableInWave
    rvals: Sequence[VariableInWave]

    def __str__(self) -> str:
        return f"{self.lval} =~ {' + '.join(map(str, self.rvals))}"

    def __hash__(self) -> int:
        return self.lval.__hash__() + sum((i + 1) * val.__hash__() for i, val in enumerate(self.rvals))


@dataclass(frozen=True)
class Covariance:
    lval: VariableInWave
    rvals: Sequence[VariableInWave]

    def __str__(self) -> str:
        return f"{self.lval} ~~ {' + '.join(map(str, self.rvals))}"

    def __hash__(self) -> int:
        return self.lval.__hash__() + sum((i + 1) * val.__hash__() for i, val in enumerate(self.rvals))


class OrdinalVariableSet(set[VariableInWave]):
    def build(self) -> str:
        if len(self) == 0:
            return ""

        return f"DEFINE(ordinal) {' '.join(sorted(str(variable) for variable in self))}"


class ModelDefinitionBuilder:
    y: VariableDefinition
    y_lag_structure: list[int]

    x: VariableDefinition
    x_lag_structure: list[int]

    w: list[VariableDefinition] | None = None

    def __init__(self):
        self._regressions: list[Regression] = []
        self._measurements: list[Measurement] = []
        self._covariances: list[Covariance] = []
        self._ordinals = OrdinalVariableSet()

        self.w = []

    def with_y(self, y: VariableDefinition, *, lag_structure: list[int] | None = None) -> Self:
        if y.is_dummy:
            raise NotImplementedError("Cannot have dependent variable be a dummy variable, must be interval scale.")

        self.y = y

        if lag_structure is not None:
            # Zero lag breaks the regression, negative lags break the logic for when the first/last regressions are
            assert all(i > 0 for i in lag_structure) and len(lag_structure) == len(set(lag_structure)), (
                "Invalid lags provided for y"
            )
            self.y_lag_structure = lag_structure
        else:
            self.y_lag_structure = []

        return self

    def with_x(self, x: VariableDefinition, *, lag_structure: list[int] | None = None) -> Self:
        self.x = x

        if lag_structure is not None:
            # Negative lags break the logic for when the first/last regressions are
            assert all(i >= 0 for i in lag_structure) and len(lag_structure) == len(set(lag_structure)), (
                "Invalid lags provided for x"
            )
            self.x_lag_structure = lag_structure
        else:
            self.x_lag_structure = [0]

        return self

    def with_w(self, variables: list[VariableDefinition], ordinal: list[bool] | None = None) -> Self:
        self.w = variables
        self.w_ordinal = ordinal if ordinal is not None else [False] * len(variables)
        return self

    def build(self, available_columns: "pd.Index[str]") -> str:
        # (name, wave) pairs in the data
        available_variables = [AvailableVariable.from_column_name(column) for column in available_columns]

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

    def _determine_start_and_end_years(self, available_variables: list[AvailableVariable]) -> Tuple[int, int]:
        y_years = [variable.wave for variable in available_variables if variable.name == self.y.name]
        y_start = min(y_years)
        y_end = max(y_years)

        x_years = [variable.wave for variable in available_variables if variable.name == self.x.name]
        x_start = min(x_years)
        x_end = max(x_years)

        # TODO?: When using FIML/handling missing data, perhaps the `max` and `min` in these two statements should swap,
        # but then we have to also add those columns to the df to prevent index errors.
        # If x goes far enough back, the first regression is when y starts
        # else, start as soon as we can due to x
        max_y_lag = max(self.y_lag_structure) if len(self.y_lag_structure) > 0 else 0
        first_year_y = max(y_start + max_y_lag, x_start + max(self.x_lag_structure))
        # If x does not go far enough forward, stop when x_stops
        # else, stop when y stops
        last_year_y = min(x_end + min(self.x_lag_structure), y_end)

        return first_year_y, last_year_y

    def _build_regressions(self, available_variables: list[AvailableVariable], first_year_y: int, last_year_y: int):
        for year_y in range(first_year_y, last_year_y + 1):
            y = VariableInWave(self.y, year_y)

            available_variable_names = [variable.name for variable in available_variables]

            if y.variable.name not in available_variable_names:
                warnings.warn(f"{y=} not found in data, skipping regression", stacklevel=2)
                continue

            y_lags = [VariableWithNamedParameter(self.y, year_y - lag, f"rho{lag}") for lag in self.y_lag_structure]
            x_lags = [VariableWithNamedParameter(self.x, year_y - lag, f"beta{lag}") for lag in self.x_lag_structure]

            w = (
                [VariableWithNamedParameter(name, year_y, f"delta0_{j}") for j, name in enumerate(self.w)]
                if self.w is not None
                else []
            )

            if (
                any(variable.variable.name not in available_variable_names for variable in y_lags)
                or any(variable.variable.name not in available_variable_names for variable in x_lags)
                or any(variable.variable.name not in available_variable_names for variable in w)
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
            if self.y.is_ordinal:
                self._ordinals.update(y_lags)

            if self.x.is_ordinal:
                self._ordinals.update(x_lags)

            w_ordinal = self.w_ordinal if self.w_ordinal is not None else []
            for variable, is_ordinal in zip(w, w_ordinal, strict=True):
                if is_ordinal:
                    self._ordinals.add(variable)

            # Fix variance for y to be constant in time
            self._covariances.append(Covariance(y, [VariableWithNamedParameter(y.variable, y.wave, "sigma")]))

    # Allow for pre-determined variables, i.e. arbitrary correlation between x and previous values of y
    # NOTE: The very first value of y in the data is considered exogenous and thus it can't be correlated with future x
    def _make_x_predetermined(self):
        # Establish list of used regressors, as defining covariances between y and unused x is meaningless
        # (and causes the model to crash)
        all_regressors: list[VariableInWave] = []
        for regression in self._regressions:
            all_regressors.extend(rval for rval in regression.rvals if rval not in all_regressors)

        for regression in self._regressions:
            y_current = regression.lval
            x_future = [
                VariableWithNamedParameter(variable.variable, variable.wave, f"gamma{variable.wave - regression.wave}")
                for variable in all_regressors
                if variable.variable == self.x and variable.wave > regression.wave
            ]

            if len(x_future) != 0:
                self._covariances.append(Covariance(y_current, x_future))
