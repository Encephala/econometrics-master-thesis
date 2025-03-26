from typing import Self, Tuple, Collection
from dataclasses import dataclass, field
import warnings

import pandas as pd

from .data.processing import cleanup_dummy


@dataclass(frozen=True)
class VariableDefinition:
    "A conceptual variable in the model."

    name: str
    is_ordinal: bool = field(default=False, kw_only=True)
    dummy_levels: Collection[str] | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class VariableInWave:
    "A variable in the dataset, that is, a specific (level of a) variable in a specific wave."

    name: str
    wave: int
    dummy_level: str | None = field(default=None, kw_only=True)

    def build(self) -> str:
        if self.dummy_level is not None:
            return f"{self.name}_{self.wave}.{cleanup_dummy(self.dummy_level)}"

        return f"{self.name}_{self.wave}"

    def _equals(self, available_variable: "AvailableVariable") -> bool:
        if self.dummy_level is None:
            return self.name == available_variable.name and self.wave == available_variable.wave

        return (
            self.name == available_variable.name
            and self.wave == available_variable.wave
            and self.dummy_level == available_variable.dummy_level
        )

    def is_in(self, available_variables: list["AvailableVariable"]) -> bool:
        return any(self._equals(variable) for variable in available_variables)

    def _to_column_name(self) -> str:
        if self.dummy_level is not None:
            return f"{self.name}_{self.wave}.{self.dummy_level}"

        return f"{self.name}_{self.wave}"

    def has_zero_variance_in(self, data: pd.DataFrame) -> bool:
        return data[self._to_column_name()].var() == 0


@dataclass(frozen=True)
class VariableWithNamedParameter(VariableInWave):
    "A variable in the dataset that has an associated named parameter."

    parameter: str = field(repr=False)

    def build(self) -> str:
        if self.dummy_level is not None:
            return f"{self.parameter}.{cleanup_dummy(self.dummy_level)}*{super().build()}"

        return f"{self.parameter}*{super().build()}"


@dataclass(frozen=True)
class AvailableVariable:
    "A variable as available in the data, effectively a parsed column name."

    name: str
    wave: int | None = None  # None for time-invariants
    dummy_level: str | None = None

    @classmethod
    def from_column_name(cls, column: str) -> Self:
        index_underscore = column.rfind("_")

        if index_underscore == -1:
            return cls(column)

        name = column[: column.rfind("_")]

        has_dummy_level = column.find(".") != -1

        if not has_dummy_level:
            try:
                wave = int(column[column.rfind("_") + 1 :])
            except ValueError:  # Couldn't parse an int
                wave = None

            return cls(name, wave)

        try:
            wave = int(column[column.rfind("_") + 1 : column.find(".")])
        except ValueError:  # Couldn't parse an int
            wave = None

        dummy_level = column[column.find(".") + 1 :]

        return cls(name, wave, dummy_level)


@dataclass(frozen=True)
class Regression:
    lval: VariableInWave
    rvals: Collection[VariableInWave]
    include_constant: bool

    def build(self) -> str:
        return (
            f"{self.lval.build()}"
            " ~ "
            f"{'alpha*1 + ' if self.include_constant is not None and self.include_constant else ''}"
            f"{' + '.join(rval.build() for rval in self.rvals)}"
        )


@dataclass(frozen=True)
class Measurement:
    lval: VariableInWave
    rvals: Collection[VariableInWave]

    def build(self) -> str:
        return f"{self.lval.build()} =~ {' + '.join(rval.build() for rval in self.rvals)}"


@dataclass(frozen=True)
class Covariance:
    lval: VariableInWave
    rvals: Collection[VariableInWave]

    def build(self) -> str:
        return f"{self.lval.build()} ~~ {' + '.join(rval.build() for rval in self.rvals)}"


class OrdinalVariableSet(set[VariableInWave]):
    def build(self) -> str:
        if len(self) == 0:
            return ""

        return f"DEFINE(ordinal) {' '.join(sorted(variable.build() for variable in self))}"


class ModelDefinitionBuilder:
    y: VariableDefinition
    y_lag_structure: list[int]

    x: VariableDefinition
    x_lag_structure: list[int]

    w: list[VariableDefinition] | None = None
    include_constant: bool = False

    def __init__(self):
        self._regressions: list[Regression] = []
        self._measurements: list[Measurement] = []  # Unused (for now)
        self._covariances: list[Covariance] = []
        self._ordinals = OrdinalVariableSet()

        self.w = []

    def with_y(self, y: VariableDefinition, *, lag_structure: list[int] | None = None) -> Self:
        if y.dummy_levels is not None:
            raise ValueError("Cannot have dependent variable be a dummy variable, must be interval scale.")  # noqa: TRY003

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
        if x.dummy_levels is not None:
            raise NotImplementedError("Dummy x-variables are not implemented yet")

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

    def with_w(self, variables: list[VariableDefinition]) -> Self:
        self.w = variables
        return self

    def with_constant(self) -> Self:
        self.include_constant = True
        return self

    def build(self, data: pd.DataFrame) -> str:
        # (name, wave) pairs in the data
        # Ignoring the constant, as it is included in the regressions directly,
        # and does not have an associated wave.
        available_variables = [AvailableVariable.from_column_name(column) for column in data.columns]

        first_year_y, last_year_y = self._determine_start_and_end_years(available_variables)

        self._build_regressions(data, available_variables, first_year_y, last_year_y)

        self._fix_y_variance()

        self._make_x_predetermined()

        return f"""# Regressions (structural part)
{"\n".join([*map(Regression.build, self._regressions), ""])}
# Measurement part
{"\n".join([*map(Measurement.build, self._measurements), ""])}
# Additional covariances
{"\n".join([*map(Covariance.build, self._covariances), ""])}
# Operations/constraints
{self._ordinals.build()}
"""

    def _determine_start_and_end_years(self, available_variables: list[AvailableVariable]) -> Tuple[int, int]:
        y_years = [variable.wave for variable in available_variables if variable.name == self.y.name]

        assert not any(year is None for year in y_years)  # y is never time-invariant

        y_start = min(y_years)  # pyright: ignore[reportArgumentType]
        y_end = max(y_years)  # pyright: ignore[reportArgumentType]

        x_years = [variable.wave for variable in available_variables if variable.name == self.x.name]

        assert not any(year is None for year in x_years)  # x is never time-invariant

        x_start = min(x_years)  # pyright: ignore[reportArgumentType]
        x_end = max(x_years)  # pyright: ignore[reportArgumentType]

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

    def _build_regressions(
        self,
        data: pd.DataFrame,
        available_variables: list[AvailableVariable],
        first_year_y: int,
        last_year_y: int,
    ):
        for year_y in range(first_year_y, last_year_y + 1):
            y = VariableInWave(self.y.name, year_y)

            if not y.is_in(available_variables):
                warnings.warn(f"{y=} not found in data, skipping regression", stacklevel=2)
                continue

            y_lags = [
                VariableWithNamedParameter(self.y.name, year_y - lag, f"rho{lag}") for lag in self.y_lag_structure
            ]
            x_lags = [
                VariableWithNamedParameter(self.x.name, year_y - lag, f"beta{lag}") for lag in self.x_lag_structure
            ]

            w = self._compile_w(year_y)
            rvals: list[VariableInWave] = [*y_lags, *x_lags, *w]

            # Check for missing variables
            if (missing_info := self._find_missing_variables(rvals, available_variables)) is not None:
                missing_variables, is_dummy = missing_info

                if not is_dummy:
                    # There's only one VariableInWave in missing_variables if it's not a dummy
                    warnings.warn(
                        f"For {y=}, {missing_variables[0]} was not in the data, skipping regression",
                        stacklevel=2,
                    )
                    # TODO: Is there a better option than ditching the whole regression
                    # because one of the vars is missing?
                    continue

                warnings.warn(
                    f"For {y=}, the dummy levels {missing_variables} were not in the data, excluding from regression",
                    stacklevel=2,
                )

                # Exclude the dummy level from the regression, as it isn't in the data.
                for variable in missing_variables:
                    rvals.remove(variable)

            # Check for variables with zero variance
            if (zero_variance_variables := self._find_zero_variance_variables(rvals, data)) is not None:
                warnings.warn(
                    f"For {y=}, the variables {zero_variance_variables} had zero variance, excluding from regression",
                    stacklevel=2,
                )

                for variable in zero_variance_variables:
                    rvals.remove(variable)

            self._regressions.append(Regression(y, rvals, self.include_constant))

            # TODO: Might be nicer to have this in a separate function,
            # but since it so heavily relies on local variables, leaving it here for now.
            if self.y.is_ordinal:
                self._ordinals.update(y_lags)

            if self.x.is_ordinal:
                self._ordinals.update(x_lags)

            ordinal_w = [] if self.w is None else [definition for definition in self.w if definition.is_ordinal]

            for definition in ordinal_w:
                variables = [variable for variable in w if variable.name == definition.name]
                self._ordinals.update(variables)

    def _compile_w(self, year_y: int) -> list[VariableWithNamedParameter]:
        if self.w is None:
            return []

        result = []

        for variable in self.w:
            if variable.dummy_levels is None:
                result.append(VariableWithNamedParameter(variable.name, year_y, f"delta0_{variable.name}"))
                continue

            result.extend(
                [
                    VariableWithNamedParameter(variable.name, year_y, f"delta0_{variable.name}", dummy_level=level)
                    for level in variable.dummy_levels
                ]
            )

        return result

    def _find_missing_variables(
        self,
        variables: Collection[VariableInWave],
        available_variables: list[AvailableVariable],
    ) -> Tuple[list[VariableInWave], bool] | None:
        """Checks if all the regressors are in the data.

        If a variable is missing, returns ([variable], False).
        If just a dummy level of a variable is missing, returns ([levels], True)
        Returns `None` if all variables are found."""
        missing_dummies = []

        for variable in variables:
            if not variable.is_in(available_variables):
                if variable.dummy_level is None:
                    return [variable], False

                missing_dummies.append(variable)

        # TODO: If all dummy levels for a variable are missing, this is a missing variable altogether
        # and should return False to raise an error.
        if len(missing_dummies) != 0:
            return missing_dummies, True

        return None

    def _find_zero_variance_variables(
        self,
        variables: Collection[VariableInWave],
        data: pd.DataFrame,
    ) -> list[VariableInWave] | None:
        """Checks if all the regressors are in the data.

        If a variable is missing, returns ([variable], False).
        If just a dummy level of a variable is missing, returns ([levels], True)
        Returns `None` if all variables are found."""
        zero_variance = [variable for variable in variables if variable.has_zero_variance_in(data)]

        if len(zero_variance) != 0:
            return zero_variance

        return None

    def _fix_y_variance(self):
        for regression in self._regressions:
            # Fix variance for y to be constant in time
            self._covariances.append(
                Covariance(
                    regression.lval, [VariableWithNamedParameter(regression.lval.name, regression.lval.wave, "sigma")]
                )
            )

    # Allow for pre-determined variables, i.e. arbitrary correlation between x and previous values of y
    # NOTE: The very first value of y in the data is considered exogenous and thus it can't be correlated with future x
    def _make_x_predetermined(self):
        # Establish list of used regressors, as defining covariances between y and unused x is meaningless
        # (and causes the model to crash)
        all_regressors = self._all_regressors()

        for regression in self._regressions:
            y_current = regression.lval
            x_future = [
                VariableWithNamedParameter(variable.name, variable.wave, f"gamma{variable.wave - regression.lval.wave}")
                for variable in all_regressors
                if variable.name == self.x.name and variable.wave > regression.lval.wave
            ]

            if len(x_future) != 0:
                self._covariances.append(Covariance(y_current, x_future))

    def _all_regressors(self) -> list[VariableInWave]:
        all_regressors: list[VariableInWave] = []

        for regression in self._regressions:
            all_regressors.extend(rval for rval in regression.rvals if rval not in all_regressors)

        return all_regressors
