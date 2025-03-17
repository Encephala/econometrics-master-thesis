from itertools import chain
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

    def __str__(self) -> str:
        return f"{self.variable}_{self.wave}"

    # For the name without any potential named parameter as in the subclass below.
    def full_name(self) -> str:
        # Hardcoded class rather than self because self.__str__ dynamically dispatches
        # to potentially overridden versions of dunder str
        return VariableInWave.__str__(self)


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

    @classmethod
    def from_column_name(cls, column: str) -> Self:
        index_underscore = column.rfind("_")
        if index_underscore == -1:
            raise ValueError(f'Column "{column}" is not in wide format (<variable>_<year>).')  # noqa: TRY003

        name = column[: column.rfind("_")]

        has_dummy_level = column.find("|") != -1

        if has_dummy_level:
            wave = int(column[column.rfind("_") + 1 : column.find("|")])
        else:
            wave = int(column[column.rfind("_") + 1 :])

        return cls(name, wave)

    def full_name(self) -> str:
        return f"{self.name}_{self.wave}"


@dataclass(frozen=True)
class Regression:
    lval: VariableInWave
    rvals: Sequence[VariableInWave]
    constant_name: str | None = field(default=None)

    def __str__(self) -> str:
        return (
            f"{self.lval}"
            " ~ "
            f"{f'alpha*{self.constant_name} + ' if self.constant_name is not None else ''}"
            f"{' + '.join(map(str, self.rvals))}"
        )


@dataclass(frozen=True)
class Measurement:
    lval: VariableInWave
    rvals: Sequence[VariableInWave]

    def __str__(self) -> str:
        return f"{self.lval} =~ {' + '.join(map(str, self.rvals))}"


@dataclass(frozen=True)
class Covariance:
    lval: VariableInWave
    rvals: Sequence[VariableInWave]

    def __str__(self) -> str:
        return f"{self.lval} ~~ {' + '.join(map(str, self.rvals))}"


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
    constant: str | None = None

    def __init__(self):
        self._regressions: list[Regression] = []
        self._measurements: list[Measurement] = []  # Unused (for now)
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

    def with_constant(self, constant_name: str) -> Self:
        self.constant = constant_name
        return self

    def build(self, available_columns: "pd.Index[str]") -> str:
        # (name, wave) pairs in the data
        # Ignoring the constant, as it is included in the regressions directly,
        # and does not have an associated wave.
        available_variables = [
            AvailableVariable.from_column_name(column)
            for column in available_columns.drop(self.constant, errors="ignore")
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

            available_variable_names = [variable.full_name() for variable in available_variables]

            if y.full_name() not in available_variable_names:
                warnings.warn(f"{y=} not found in data, skipping regression", stacklevel=2)
                continue

            y_lags = [VariableWithNamedParameter(self.y, year_y - lag, f"rho{lag}") for lag in self.y_lag_structure]
            x_lags = [VariableWithNamedParameter(self.x, year_y - lag, f"beta{lag}") for lag in self.x_lag_structure]

            w = (
                [VariableWithNamedParameter(name, year_y, f"delta0_{j}") for j, name in enumerate(self.w)]
                if self.w is not None
                else []
            )

            if (missing_var := self._find_missing_variables(available_variable_names, y_lags, x_lags, w)) is not None:
                warnings.warn(
                    f"For {y=}, {missing_var.full_name()} was not in the data, skipping regression",
                    stacklevel=2,
                )
                # TODO: Is there a better option than ditching the whole regression because one of the vars is missing?
                continue

            rvals = [*y_lags, *x_lags, *w]
            self._regressions.append(Regression(y, rvals, constant_name=self.constant))

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

    def _find_missing_variables(
        self,
        available_variables_names: list[str],
        y_lags: Sequence[VariableInWave],
        x_lags: Sequence[VariableInWave],
        w: Sequence[VariableInWave],
    ) -> VariableInWave | None:
        """Checks if all the regressors are in the data.

        If one is missing, returns it.
        Returns `None` if all variables are found."""
        for variable in chain(y_lags, x_lags, w):
            if variable.full_name() not in available_variables_names:
                return variable

        return None

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
                VariableWithNamedParameter(
                    variable.variable, variable.wave, f"gamma{variable.wave - regression.lval.wave}"
                )
                for variable in all_regressors
                if variable.variable == self.x and variable.wave > regression.lval.wave
            ]

            if len(x_future) != 0:
                self._covariances.append(Covariance(y_current, x_future))
