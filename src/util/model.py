from typing import Self, Tuple, Collection
from dataclasses import dataclass, field
import logging

import pandas as pd

from .data import Column, assert_column_type_correct, cleanup_dummy, find_non_PD_suspicious_columns

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariableDefinition:
    "A conceptual variable in the model."

    name: str
    is_ordinal: bool = field(default=False, kw_only=True)
    dummy_levels: Collection[str] | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class Variable:
    "A variable in the dataset, that is, a specific (level of a) variable in a specific wave."

    name: str
    wave: int | None
    dummy_level: str | None = field(default=None, kw_only=True)

    def build(self) -> str:
        result = f"{self.name}"

        if self.wave is not None:
            result += f"_{self.wave}"

        if self.dummy_level is not None:
            result += f".{self.dummy_level}"

        return result

    def _equals(self, other: "Column") -> bool:
        match (self.wave, self.dummy_level):
            case (None, None):
                return self.name == other.name
            case (wave, None):
                return self.name == other.name and wave == other.wave
            case (None, dummy_level):
                return self.name == other.name and dummy_level == other.dummy_level
            case (wave, dummy_level):
                return self.name == other.name and wave == other.wave and dummy_level == other.dummy_level

    def is_in(self, available_variables: Collection["Column"]) -> bool:
        return any(self._equals(variable) for variable in available_variables)

    def has_zero_variance_in(self, data: pd.DataFrame) -> bool:
        return data[Column(self.name, self.wave, self.dummy_level)].var() == 0


@dataclass(frozen=True)
class VariableWithNamedParameter(Variable):
    "A variable in the dataset that has an associated named parameter."

    parameter: str = field(repr=False)

    def build(self) -> str:
        if self.dummy_level is not None:
            return f"{self.parameter}.{cleanup_dummy(self.dummy_level)}*{super().build()}"

        return f"{self.parameter}*{super().build()}"


@dataclass(frozen=True)
class Regression:
    lval: Variable
    rvals: Collection[Variable]
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
    lval: Variable
    rvals: Collection[Variable]

    def build(self) -> str:
        return f"{self.lval.build()} =~ {' + '.join(rval.build() for rval in self.rvals)}"


@dataclass(frozen=True)
class Covariance:
    lval: Variable
    rvals: Collection[Variable]

    def build(self) -> str:
        return f"{self.lval.build()} ~~ {' + '.join(rval.build() for rval in self.rvals)}"


class OrdinalVariableSet(set[Variable]):
    def build(self) -> str:
        if len(self) == 0:
            return ""

        return f"DEFINE(ordinal) {' '.join(sorted(variable.build() for variable in self))}"


class ModelDefinitionBuilder:
    _y: VariableDefinition
    _y_lag_structure: list[int]

    _x: VariableDefinition
    _x_lag_structure: list[int]

    _w: list[VariableDefinition] | None = None
    _include_constant: bool = False

    _do_missing_check: bool = False
    _do_variance_check: bool = False
    _do_PD_check: bool = False

    # To make covariance PD
    _excluded_regressors: list[Column]

    # Internals
    _regressions: list[Regression]
    _measurements: list[Measurement]  # Unused (for now)
    _covariances: list[Covariance]
    _ordinals: OrdinalVariableSet

    def __init__(self):
        self._regressions = []
        self._measurements = []  # Unused (for now)
        self._covariances = []
        self._ordinals = OrdinalVariableSet()

        self._w = []
        self._excluded_regressors = []

    def with_y(self, y: VariableDefinition, *, lag_structure: list[int] | None = None) -> Self:
        if y.dummy_levels is not None:
            raise ValueError("Cannot have dependent variable be a dummy variable, must be interval scale.")  # noqa: TRY003

        self._y = y

        if lag_structure is not None:
            # Zero lag breaks the regression, negative lags break the logic for when the first/last regressions are
            assert all(i > 0 for i in lag_structure) and len(lag_structure) == len(set(lag_structure)), (
                "Invalid lags provided for y"
            )
            self._y_lag_structure = lag_structure
        else:
            self._y_lag_structure = []

        return self

    def with_x(self, x: VariableDefinition, *, lag_structure: list[int] | None = None) -> Self:
        if x.dummy_levels is not None:
            raise NotImplementedError("Dummy x-variables are not implemented yet")

        self._x = x

        if lag_structure is not None:
            # Negative lags break the logic for when the first/last regressions are
            assert all(i >= 0 for i in lag_structure) and len(lag_structure) == len(set(lag_structure)), (
                "Invalid lags provided for x"
            )
            self._x_lag_structure = lag_structure
        else:
            self._x_lag_structure = [0]

        return self

    def with_w(self, variables: list[VariableDefinition]) -> Self:
        self._w = variables
        return self

    def with_constant(self) -> Self:
        self._include_constant = True
        return self

    def with_checks(self, *, missing: bool = True, variance: bool = True, PD: bool = True) -> Self:
        self._do_missing_check = missing
        self._do_variance_check = variance
        self._do_PD_check = PD
        return self

    def with_excluded_regressors(self, excluded_regressors: list[Column]) -> Self:
        self._excluded_regressors = excluded_regressors
        return self

    def build(self, data: pd.DataFrame) -> str:
        assert_column_type_correct(data)

        available_variables: list[Column] = list(data.columns)  # pyright: ignore[reportAssignmentType]

        waves = self._available_dependent_variables(available_variables)

        self._build_regressions(waves, available_variables, data)

        self._fix_y_variance()

        self._make_x_predetermined()

        if self._do_PD_check:
            self._check_covariance_matrix_PD(data)

        return self._make_result()

    def _available_dependent_variables(self, available_variables: list[Column]) -> list[Column]:
        y_years = [variable.wave for variable in available_variables if variable.name == self._y.name]

        assert not any(year is None for year in y_years)  # y is never time-invariant

        y_start = min(y_years)  # pyright: ignore[reportArgumentType]
        y_end = max(y_years)  # pyright: ignore[reportArgumentType]

        x_years = [variable.wave for variable in available_variables if variable.name == self._x.name]

        assert not any(year is None for year in x_years)  # x is never time-invariant

        x_start = min(x_years)  # pyright: ignore[reportArgumentType]
        x_end = max(x_years)  # pyright: ignore[reportArgumentType]

        # TODO?: When using FIML/handling missing data, perhaps the `max` and `min` in these two statements should swap,
        # but then we have to also add those columns to the df to prevent index errors.
        # If x goes far enough back, the first regression is when y starts
        # else, start as soon as we can due to x
        max_y_lag = max(self._y_lag_structure) if len(self._y_lag_structure) > 0 else 0
        first_year_y = max(y_start + max_y_lag, x_start + max(self._x_lag_structure))
        # If x does not go far enough forward, stop when x_stops
        # else, stop when y stops
        last_year_y = min(x_end + min(self._x_lag_structure), y_end)

        return [
            variable
            for variable in available_variables
            if variable.name == self._y.name and first_year_y <= variable.wave <= last_year_y
        ]

    def _build_regressions(
        self,
        dependent_vars: list[Column],
        available_variables: Collection[Column],
        data: pd.DataFrame,
    ):
        for variable in dependent_vars:
            y = Variable(variable.name, variable.wave)

            wave: int = y.wave  # pyright: ignore[reportAssignmentType]

            y_lags = [
                VariableWithNamedParameter(self._y.name, wave - lag, f"rho{lag}") for lag in self._y_lag_structure
            ]
            x_lags = [
                VariableWithNamedParameter(self._x.name, wave - lag, f"beta{lag}") for lag in self._x_lag_structure
            ]

            w = self._compile_w(wave)
            rvals: list[Variable] = [*y_lags, *x_lags, *w]

            rvals = self._remove_excluded_regressors(rvals)

            if self._do_missing_check:
                rvals = self._filter_missing_rvals(y, rvals, available_variables)

            if self._do_variance_check:
                rvals = self._filter_constant_rvals(y, rvals, data)

            self._regressions.append(Regression(y, rvals, self._include_constant))

            # TODO: Might be nicer to have this in a separate function,
            # but since it so heavily relies on local variables, leaving it here for now.
            if self._y.is_ordinal:
                self._ordinals.update(y_lags)

            if self._x.is_ordinal:
                self._ordinals.update(x_lags)

            ordinal_w = [] if self._w is None else [definition for definition in self._w if definition.is_ordinal]

            for definition in ordinal_w:
                variables = [variable for variable in w if variable.name == definition.name]
                self._ordinals.update(variables)

    def _compile_w(self, wave_y: int | None) -> list[VariableWithNamedParameter]:
        if self._w is None:
            return []

        result = []

        for variable in self._w:
            if variable.dummy_levels is None:
                result.append(VariableWithNamedParameter(variable.name, wave_y, f"delta0_{variable.name}"))
                continue

            result.extend(
                [
                    VariableWithNamedParameter(variable.name, wave_y, f"delta0_{variable.name}", dummy_level=level)
                    for level in variable.dummy_levels
                ]
            )

        return result

    def _remove_excluded_regressors(self, rvals: list[Variable]) -> list[Variable]:
        return [rval for rval in rvals if not rval.is_in(self._excluded_regressors)]

    def _filter_missing_rvals(
        self, y: Variable, rvals: list[Variable], available_variables: Collection[Column]
    ) -> list[Variable]:
        # Check for missing variables
        if (missing_info := self._find_missing_variables(rvals, available_variables)) is not None:
            missing_variables, is_dummy = missing_info

            if not is_dummy:
                # There's only one Variable in missing_variables if it's not a dummy
                logger.warning(f"For {y=}, {missing_variables[0]} was not in the data, skipping regression")
                # TODO: Is there a better option than ditching the whole regression
                # because one of the vars is missing?

            logger.warning(
                f"For {y=}, the dummy levels {missing_variables} were not in the data, excluding from regression"
            )

            # Exclude the dummy level from the regression, as it isn't in the data.
            for variable in missing_variables:
                rvals.remove(variable)

        return rvals

    def _filter_constant_rvals(self, y: Variable, rvals: list[Variable], data: pd.DataFrame) -> list[Variable]:
        # Check for variables with zero variance
        if (zero_variance_variables := self._find_zero_variance_variables(rvals, data)) is not None:
            logger.warning(
                f"For {y=}, the variables {zero_variance_variables} had zero variance, excluding from regression"
            )

            for variable in zero_variance_variables:
                rvals.remove(variable)

        return rvals

    def _find_missing_variables(
        self,
        variables: Collection[Variable],
        available_variables: Collection[Column],  # pyright: ignore[reportInvalidTypeArguments]
    ) -> Tuple[list[Variable], bool] | None:
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
        variables: Collection[Variable],
        data: pd.DataFrame,
    ) -> list[Variable] | None:
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
                VariableWithNamedParameter(variable.name, variable.wave, f"gamma{variable.wave - regression.lval.wave}")  # pyright: ignore[reportOperatorIssue]
                for variable in all_regressors
                if variable.name == self._x.name and variable.wave > regression.lval.wave  # pyright: ignore[reportOperatorIssue]
            ]

            if len(x_future) != 0:
                self._covariances.append(Covariance(y_current, x_future))

    def _all_regressors(self) -> list[Variable]:
        all_regressors: list[Variable] = []

        for regression in self._regressions:
            all_regressors.extend(rval for rval in regression.rvals if rval not in all_regressors)

        return all_regressors

    def _check_covariance_matrix_PD(self, data: pd.DataFrame):
        all_regressors = [
            Column(variable.name, variable.wave, variable.dummy_level) for variable in self._all_regressors()
        ]

        subset = data[all_regressors]

        suspicious_columns = find_non_PD_suspicious_columns(subset)

        if len(suspicious_columns) > 0:
            logger.warning(f"Data covariance matrix is not PD, culprits seem to be {suspicious_columns}")

    def _make_result(self) -> str:
        return f"""# Regressions (structural part)
{"\n".join([*map(Regression.build, self._regressions), ""])}
# Measurement part
{"\n".join([*map(Measurement.build, self._measurements), ""])}
# Additional covariances
{"\n".join([*map(Covariance.build, self._covariances), ""])}
# Operations/constraints
{self._ordinals.build()}
"""

    def build_nonpanel(self, data: pd.DataFrame) -> str:
        assert_column_type_correct(data)

        available_variables: list[Column] = list(data.columns)  # pyright: ignore[reportAssignmentType]

        self._build_nonpanel_regression(data, available_variables)

        if self._do_PD_check:
            self._check_covariance_matrix_PD(data)

        return self._make_result()

    def _build_nonpanel_regression(self, data: pd.DataFrame, available_variables: list[Column]):
        # Majorly copy-pasta'd from self._build_regressions

        y = Variable(self._y.name, None)
        assert self._y_lag_structure == [], (
            f"Building nonpanel regression but y had lag structure {self._y_lag_structure}"
        )

        x = Variable(self._x.name, None)
        assert self._x_lag_structure in ([0], []), (
            f"Building nonpanel regression but x had lag structure {self._x_lag_structure}"
        )

        # Apparently [*<values>] typecasts to parent type?
        w = self._compile_w(None)
        rvals: list[Variable] = [x, *w]

        rvals = self._remove_excluded_regressors(rvals)

        if self._do_missing_check:
            rvals = self._filter_missing_rvals(y, rvals, available_variables)

        if self._do_variance_check:
            rvals = self._filter_constant_rvals(y, rvals, data)

        self._regressions.append(Regression(y, rvals, self._include_constant))

        # TODO: Might be nicer to have this in a separate function,
        # but since it so heavily relies on local variables, leaving it here for now.
        if self._y.is_ordinal:
            self._ordinals.add(y)

        if self._x.is_ordinal:
            self._ordinals.add(x)

        ordinal_w = [] if self._w is None else [definition for definition in self._w if definition.is_ordinal]

        for definition in ordinal_w:
            variables = [variable for variable in w if variable.name == definition.name]
            self._ordinals.update(variables)
