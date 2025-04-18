from typing import Self, Tuple, Sequence
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from itertools import groupby

import pandas as pd

from .data import Column
from .data.util import assert_column_type_correct, cleanup_dummy, find_non_PD_suspicious_columns

logger = logging.getLogger(__name__)


@dataclass
class VariableDefinition:
    "A conceptual variable in the model."

    name: str
    is_ordinal: bool = field(default=False, kw_only=True)
    dummy_levels: Sequence[str] | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.dummy_levels is not None:
            assert len(self.dummy_levels) != 0, "If VariableDefinition.dummy_levels is not None, it must not be empty."

            self.dummy_levels = [cleanup_dummy(level) for level in self.dummy_levels]


@dataclass(frozen=True)
class Variable:
    "A variable in the dataset, that is, a specific (level of a) variable in a specific wave."

    name: str
    wave: int | None = field(default=None, kw_only=True)
    dummy_level: str | None = field(default=None, kw_only=True)

    def build(self) -> str:
        result = f"{self.name}"

        if self.wave is not None:
            result += f"_{self.wave}"

        if self.dummy_level is not None:
            result += f".{self.dummy_level}"

        return result

    def equals(self, other: "Column") -> bool:
        result = self.name == other.name

        if self.wave is not None:
            result = result and self.wave == other.wave

        if self.dummy_level is not None:
            result = result and self.dummy_level == other.dummy_level

        return result

    def is_in(self, available_variables: Sequence["Column"]) -> bool:
        return any(self.equals(variable) for variable in available_variables)

    def has_zero_variance_in(self, data: pd.DataFrame) -> bool:
        return data[Column(self.name, self.wave, self.dummy_level)].var() == 0

    def with_named_parameter(self, parameter: str) -> "VariableWithNamedParameter":
        return VariableWithNamedParameter(self.name, wave=self.wave, dummy_level=self.dummy_level, parameter=parameter)

    def to_unnamed(self) -> "Variable":
        return Variable(self.name, wave=self.wave, dummy_level=self.dummy_level)

    def as_parameter_name(self) -> str:
        if self.dummy_level is None:
            return self.name

        return f"{self.name}.{self.dummy_level}"


@dataclass(frozen=True)
class VariableWithNamedParameter(Variable):
    "A variable in the dataset that has an associated named parameter."

    parameter: str = field(repr=False, kw_only=True)

    def build(self) -> str:
        return f"{self.parameter}*{super().build()}"


@dataclass(frozen=True)
class Regression:
    lval: Variable
    rvals: Sequence[Variable]
    include_time_dummy: bool

    def build(self) -> str:
        return (
            f"{self.lval.build()}"
            + " ~ "
            + (f"alpha_t{self.lval.wave}*1 + " if self.include_time_dummy else "")
            + (" + ".join(rval.build() for rval in self.rvals))
        )


@dataclass(frozen=True)
class Covariance:
    lval: Variable
    rvals: Sequence[Variable]

    def build(self) -> str:
        return f"{self.lval.build()} ~~ {' + '.join(rval.build() for rval in self.rvals)}"


class OrdinalVariableSet(set[Variable]):
    def build(self) -> str:
        if len(self) == 0:
            return ""

        return f"DEFINE(ordinal) {' '.join(sorted(variable.build() for variable in self))}"


@dataclass(frozen=True)
class MediatorPathway:
    mediator: Variable

    main_param: str
    mediator_params: list[str]

    for_dummy_level: str | None = None

    def build(self) -> str:
        return " + ".join([f"{self.main_param}*{mediator_param}" for mediator_param in self.mediator_params])


class _ModelDefinitionBuilder(ABC):
    _y: VariableDefinition
    _x: VariableDefinition

    _mediators: list[VariableDefinition]

    _controls: list[VariableDefinition]
    _include_time_dummy: bool = False

    _do_missing_check: bool = True
    _do_variance_check: bool = True
    _do_PD_check: bool = True

    # For identification
    _excluded_regressors: list[Column]

    # To set covariances between dummy levels of the same variable as free parameters
    _do_add_dummy_covariances: bool = False

    # Internals
    _regressions: list[Regression]
    _covariances: list[Covariance]
    _ordinals: OrdinalVariableSet
    _mediator_pathways: list[MediatorPathway]

    def __init__(self):
        self._mediators = []
        self._controls = []
        self._excluded_regressors = []

        self._regressions = []
        self._covariances = []
        self._ordinals = OrdinalVariableSet()
        self._mediator_pathways = []

    def with_mediators(self, mediators: list[VariableDefinition]) -> Self:
        self._mediators = mediators

        self._check_duplicate_definition()

        return self

    def with_controls(self, controls: list[VariableDefinition]) -> Self:
        self._controls = controls

        self._check_duplicate_definition()

        return self

    def with_time_dummy(self) -> Self:
        self._include_time_dummy = True
        return self

    def with_checks(self, *, missing: bool = True, variance: bool = True, PD: bool = True) -> Self:
        self._do_missing_check = missing
        self._do_variance_check = variance
        self._do_PD_check = PD
        return self

    def with_excluded_regressors(self, excluded_regressors: list[Column]) -> Self:
        self._excluded_regressors = excluded_regressors
        return self

    @abstractmethod
    def with_additional_covariances(self) -> Self: ...

    def _check_duplicate_definition(self):
        all_regressors = [self._x, *self._mediators, *self._controls]

        seen_regressors: set[str] = set()

        for regressor in all_regressors:
            if regressor.name in seen_regressors:
                raise ValueError(f"Duplicate use of {regressor} as rval")  # noqa: TRY003

            seen_regressors.add(regressor.name)

    @abstractmethod
    def build(self, data: pd.DataFrame, *, drop_first_dummy: bool = True) -> str: ...

    def _filter_excluded_regressors(self, rvals: list[Variable]) -> list[Variable]:
        result = []

        # For debug warning about requested excluded regressors that weren't in the model in the first place
        removed_variables: list[Variable] = []

        for rval in rvals:
            if not rval.is_in(self._excluded_regressors):
                result.append(rval)

            else:
                removed_variables.append(rval)
                logger.debug(f"Removing {rval} from regressors as requested")

        for regressor in self._excluded_regressors:
            for variable in removed_variables:
                if variable.equals(regressor):
                    # It was removed, continue outer loop
                    break

            else:
                # TODO: This still fires if the requested regressor is a dummy level that was automatically excluded
                # in self._compile_regressors.
                # It's not a solution to run this method in _compile_regressors though,
                # as at that point we don't know the regressors yet.
                # TODO: It also fires for every wave in a panel regression, that doesn't make sense.
                logger.debug(f"{regressor} was requested to be removed, but it wasn't in the model")

        return result

    def _filter_missing_rvals(
        self, y: Variable, rvals: list[Variable], available_variables: Sequence[Column]
    ) -> tuple[bool, list[Variable]]:
        """Returns (skip_regression: `bool`, filtered_rvals: `list[Variable]`)."""
        # Check for missing variables
        if (missing_info := self._find_missing_variables(rvals, available_variables)) is not None:
            missing_variables, is_dummy = missing_info

            if not is_dummy:
                # There's only one Variable in missing_variables if it's not a dummy
                logger.warning(f"For {y=}, {missing_variables} were not in the data, skipping regression")
                # TODO: Is there a better option than ditching the whole regression
                # because one of the vars is missing?
                return True, []

            logger.warning(
                f"For {y=}, the dummy levels {missing_variables} were not in the data, excluding from regression"
            )

            # Exclude the dummy level from the regression, as it isn't in the data.
            for variable in missing_variables:
                rvals.remove(variable)

        return False, rvals

    def _find_missing_variables(
        self,
        model_variables: Sequence[Variable],
        available_variables: Sequence[Column],  # pyright: ignore[reportInvalidTypeArguments]
    ) -> Tuple[list[Variable], bool] | None:
        """Checks if all the regressors are in the data.

        If a variable is missing, returns ([variable], False).
        If just a dummy level of a variable is missing, returns ([levels], True)
        Returns `None` if all variables are found."""
        all_missing_dummies: list[Variable] = []

        for variable in model_variables:
            if not variable.is_in(available_variables):
                if variable.dummy_level is None:
                    return [variable], False

                all_missing_dummies.append(variable)

        if len(all_missing_dummies) == 0:
            return None

        # If all dummy levels for a variable are missing, this is a missing variable altogether
        # and we should return False to trigger an error.
        for name, model_vars in groupby(model_variables, lambda variable: variable.name):
            missing_dummies = [dummy for dummy in all_missing_dummies if dummy.name == name]

            if len(missing_dummies) == 0:
                continue

            missing_dummy_levels = [var.dummy_level for var in missing_dummies]
            model_dummy_level = [var.dummy_level for var in model_vars]

            if all((level in missing_dummy_levels) for level in model_dummy_level):
                return missing_dummies, False

        # All good, just some random dummy levels
        return all_missing_dummies, True

    def _filter_constant_rvals(self, y: Variable, rvals: list[Variable], data: pd.DataFrame) -> list[Variable]:
        # Check for variables with zero variance
        if (zero_variance_variables := self._find_zero_variance_variables(rvals, data)) is not None:
            logger.warning(
                f"For {y=}, the variables {zero_variance_variables} had zero variance, excluding from regression"
            )

            for variable in zero_variance_variables:
                rvals.remove(variable)

        return rvals

    def _find_zero_variance_variables(
        self,
        variables: Sequence[Variable],
        data: pd.DataFrame,
    ) -> list[Variable] | None:
        """Checks if all the regressors have finite variance.

        If a variable has zero variance, returns [variable].
        Returns `None` if all variables have finite variance."""
        zero_variance = [variable for variable in variables if variable.has_zero_variance_in(data)]

        if len(zero_variance) != 0:
            return zero_variance

        return None

    def _define_ordinals(
        self,
        y: Sequence[Variable],
        x: Sequence[Variable],
        mediators: Sequence[Variable],
        controls: Sequence[Variable],
    ):
        if self._y.is_ordinal:
            self._ordinals.update(y)

        if self._x.is_ordinal:
            self._ordinals.update(x)

        ordinal_mediators = [definition for definition in self._mediators if definition.is_ordinal]
        for definition in ordinal_mediators:
            variables = [variable for variable in mediators if variable.name == definition.name]
            self._ordinals.update(variables)

        ordinal_controls = [definition for definition in self._controls if definition.is_ordinal]
        for definition in ordinal_controls:
            variables = [variable for variable in controls if variable.name == definition.name]
            self._ordinals.update(variables)

    def _all_regressors(self) -> list[Variable]:
        all_regressors: list[Variable] = []

        for regression in self._regressions:
            all_regressors.extend(rval.to_unnamed() for rval in regression.rvals if rval not in all_regressors)

        # Filter non-unique, conveniently maintains order
        return list(dict.fromkeys(all_regressors))

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
{"\n".join(map(Regression.build, self._regressions))}

# Total effect
{"\n".join(self._make_total_effects())}

# Additional covariances
{"\n".join(map(Covariance.build, self._covariances))}

# Operations/constraints
{self._ordinals.build()}
"""

    def _make_total_effects(self) -> list[str]:
        if self._x.dummy_levels is None:
            return self._make_total_for_dummy_level(self._mediator_pathways, None)

        result: list[str] = []

        for x_dummy_level in self._x.dummy_levels:
            relevant_pathways = list(
                filter(lambda pathway: pathway.for_dummy_level == x_dummy_level, self._mediator_pathways)
            )

            if len(relevant_pathways) == 0:
                # This is a dummy level that was dropped for identification
                continue

            result.extend(self._make_total_for_dummy_level(relevant_pathways, x_dummy_level))

        return result

    def _make_total_for_dummy_level(
        self,
        relevant_pathways: Sequence[MediatorPathway],
        dummy_level: str | None,
    ) -> list[str]:
        def make_effect_name(prefix: str, dummy_level: str | None, suffix: str | None) -> str:
            result = prefix

            if dummy_level is not None:
                result += f"_{dummy_level}"

            if suffix is not None:
                result += f"_{suffix}"

            return result

        result: list[str] = []

        # TODO: Only valid for x_lag_structure = [0] right now
        # Not sure yet how to generalise to multiple x-lags
        direct_effect_name = make_effect_name("effect", dummy_level, "direct")

        # NOTE: Mimicking `Variable.as_parameter_name`
        # Could avoid the duplication when including x in a MediatorPathway, but not sure how
        # to handle multiple lags yet so that's a TODO
        if dummy_level is not None:
            result.append(f"{direct_effect_name} := beta0_{self._x.name}.{dummy_level}")
        else:
            result.append(f"{direct_effect_name} := beta0_{self._x.name}")

        components_global_total: list[str] = [direct_effect_name]

        for variable_name, pathways in groupby(relevant_pathways, lambda pathway: pathway.mediator.name):
            # Contributions to mediation through this entire variable
            components_local_total: list[str] = []

            for pathway in pathways:
                local_name = make_effect_name("effect", dummy_level, pathway.mediator.as_parameter_name())

                contributions = [f"{pathway.main_param}*{mediator_param}" for mediator_param in pathway.mediator_params]

                result.append(f"{local_name} := {' + '.join(contributions)}")
                components_local_total.append(local_name)

            global_component_name = make_effect_name("total", dummy_level, variable_name)
            result.append(f"{global_component_name} := {' + '.join(components_local_total)}")
            components_global_total.append(global_component_name)

        global_name = make_effect_name("total", dummy_level, None)
        result.append(f"{global_name} := {' + '.join(components_global_total)}")

        return result


class PanelModelDefinitionBuilder(_ModelDefinitionBuilder):
    _y_lag_structure: list[int]
    _x_lag_structure: list[int]

    # To make epsilon fixed
    _fix_regressand_variance: bool = True

    # To set covariance between current value and future value of a regressor as a free value
    _free_covariance_across_time: bool = True

    _do_make_x_predetermined: bool = True

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

    def with_additional_covariances(
        self,
        *,
        fixed_regressand_variance: bool = True,
        free_covariance_across_time: bool = True,
        within_dummy_covariance: bool = True,
        x_predetermined: bool = True,
    ) -> Self:
        self._fix_regressand_variance = fixed_regressand_variance
        self._free_covariance_across_time = free_covariance_across_time
        self._do_add_dummy_covariances = within_dummy_covariance
        self._do_make_x_predetermined = x_predetermined
        return self

    def build(self, data: pd.DataFrame, *, drop_first_dummy: bool = True) -> str:
        assert_column_type_correct(data)

        available_variables: list[Column] = list(data.columns)  # pyright: ignore[reportAssignmentType]

        waves = self._available_dependent_variables(available_variables)

        self._build_regressions(waves, available_variables, data, drop_first_dummy)

        if self._fix_regressand_variance:
            self._fix_variances_across_time()

        if self._free_covariance_across_time:
            self._free_regressor_correlations_across_time()

        if self._do_make_x_predetermined:
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
        available_variables: Sequence[Column],
        data: pd.DataFrame,
        drop_first_dummy: bool,  # noqa: FBT001
    ):
        for variable in dependent_vars:
            y = Variable(variable.name, wave=variable.wave)

            wave: int = y.wave  # pyright: ignore[reportAssignmentType]

            y_lags = [
                VariableWithNamedParameter(self._y.name, wave=wave - lag, parameter=f"rho{lag}")
                for lag in self._y_lag_structure
            ]

            x_lags: list[Variable] = []
            for lag in self._x_lag_structure:
                x_lags.extend(self._compile_regressors([self._x], f"beta{lag}", wave - lag, drop_first_dummy))

            mediators = self._compile_regressors(self._mediators, "zeta", wave, drop_first_dummy)

            controls = self._compile_regressors(self._controls, "delta0", wave, drop_first_dummy)
            rvals: list[Variable] = [*y_lags, *x_lags, *mediators, *controls]

            rvals = self._filter_excluded_regressors(rvals)

            if self._do_missing_check:
                skip_regression, rvals = self._filter_missing_rvals(y, rvals, available_variables)

                if skip_regression:
                    continue

            if self._do_variance_check:
                rvals = self._filter_constant_rvals(y, rvals, data)

            self._regressions.append(Regression(y, rvals, self._include_time_dummy))
            self._add_mediator_regressions(mediators, y_lags, x_lags, controls)

            if self._do_add_dummy_covariances:
                self._add_dummy_covariances(rvals)

            self._define_ordinals([y, *y_lags], [*x_lags], mediators, controls)

    def _compile_regressors(
        self,
        regressors: list[VariableDefinition],
        parameter_name: str,
        wave_y: int,
        drop_first_dummy: bool,  # noqa: FBT001
    ) -> list[Variable]:
        result = []

        for variable in regressors:
            if variable.dummy_levels is None:
                result.append(
                    VariableWithNamedParameter(
                        variable.name, wave=wave_y, parameter=f"{parameter_name}_{variable.name}"
                    )
                )
                continue

            dummy_levels = variable.dummy_levels

            if drop_first_dummy:
                dropped, *dummy_levels = dummy_levels
                logger.debug(f"Dropped first dummy level '{dropped}' for {variable.name}")

            result.extend(
                [
                    VariableWithNamedParameter(
                        variable.name,
                        wave=wave_y,
                        parameter=f"{parameter_name}_{variable.name}.{level}",
                        dummy_level=level,
                    )
                    for level in dummy_levels
                ]
            )

        return result

    def _add_mediator_regressions(
        self,
        mediators: Sequence[Variable],
        y_lags: Sequence[Variable],
        x_lags: Sequence[Variable],
        controls: Sequence[Variable],
    ):
        for mediator in [mediator.to_unnamed() for mediator in mediators]:
            current_rvals: list[VariableWithNamedParameter] = []

            for lag, lagged_y in zip(self._y_lag_structure, y_lags, strict=True):
                current_rvals.append(lagged_y.with_named_parameter(f"eta_{mediator.as_parameter_name()}_y{lag}"))

            for lag, lagged_x in zip(self._x_lag_structure, x_lags, strict=True):
                current_rvals.append(lagged_x.with_named_parameter(f"eta_{mediator.as_parameter_name()}_x{lag}"))

            for control in controls:
                current_rvals.append(  # noqa: PERF401
                    control.with_named_parameter(f"eta_{mediator.as_parameter_name()}_{control.as_parameter_name()}")
                )

            self._regressions.append(Regression(mediator, current_rvals, self._include_time_dummy))
            self._mediator_pathways.append(
                MediatorPathway(
                    mediator,
                    f"zeta_{mediator.as_parameter_name()}",
                    [f"eta_{mediator.as_parameter_name()}_x{lag}" for lag in self._x_lag_structure],
                )
            )

    def _add_dummy_covariances(self, rvals: list[Variable]):
        """Adds the covariances between dummy levels for the given rvals (lvals must be interval scale).

        Should thus be called once for each regression."""
        # NOTE/TODO: Because this function works on the rvals and not self._controls,
        # it does not include a covariance for the dummy level that is excluded for identification.
        # I'm not 100% that is correct behaviour.

        variables = groupby(rvals, lambda rval: rval.name)

        for name, values in variables:
            dummy_levels = [rval for rval in values if rval.name == name and rval.dummy_level is not None]

            if len(dummy_levels) == 0:
                continue

            for i in range(len(dummy_levels) - 1):
                lval = dummy_levels[i].to_unnamed()
                rvals = [
                    rval.with_named_parameter(f"sigma_{lval.as_parameter_name()}_{rval.as_parameter_name()}")
                    for rval in dummy_levels[i + 1 :]
                ]

                self._covariances.append(Covariance(lval, rvals))

    def _fix_variances_across_time(self):
        "Fix the variance of all the variables in self._regressions to be constant in time."
        all_variables: list[Variable] = []

        for regression in self._regressions:
            all_variables.append(regression.lval.to_unnamed())

            all_variables.extend([rval.to_unnamed() for rval in regression.rvals])

        # Remove duplicates
        all_variables = list(dict.fromkeys(all_variables))

        self._covariances.extend(
            [
                Covariance(
                    variable,
                    [variable.with_named_parameter(f"sigma_{variable.as_parameter_name()}")],
                )
                for variable in all_variables
            ]
        )

    def _free_regressor_correlations_across_time(self):
        """Sets the correlations between the same variable for different waves as a free parameter.

        NOTE: Not done for y because y's autocorrelation should be captured by the AR lags."""
        all_regressors = self._all_regressors()
        # Sort because groupby eagerly makes new groups
        all_regressors = sorted(
            all_regressors,
            key=lambda regressor: (
                regressor.name,
                regressor.wave,
                regressor.dummy_level,
            ),
        )

        for _, variables in groupby(all_regressors, lambda regressor: regressor.name):
            variables = list(variables)  # noqa: PLW2901

            # TODO: Currently, this also includes covariance with a future different dummy level
            # Would be great if that also respected _do_add_dummy_covariances
            for i in range(len(variables) - 1):
                lval = variables[i]
                rvals = [
                    rval.with_named_parameter(
                        f"gamma{rval.wave - lval.wave}_{lval.as_parameter_name()}_{rval.as_parameter_name()}"  # pyright: ignore[reportOperatorIssue]
                    )
                    for rval in variables[i + 1 :]
                    # Needed because rval might be the same wave but different dummy level
                    if rval.wave > lval.wave  # pyright: ignore[reportOperatorIssue]
                ]

                if len(rvals) != 0:
                    self._covariances.append(Covariance(lval, rvals))

    # Allow for pre-determined variables, i.e. arbitrary correlation between x and previous values of y
    def _make_x_predetermined(self):
        all_regressors = self._all_regressors()

        for regression in self._regressions:
            # Skip mediator regressions.
            # Since mediator regressions don't include x (to avoid simultaneity),
            # it's meaningless to define them as predetermined
            if regression.lval.name != self._y.name:
                continue

            y_current = regression.lval
            x_future = [
                variable.with_named_parameter(f"gamma{variable.wave - regression.lval.wave}_y_x")  # pyright: ignore[reportOperatorIssue]
                for variable in all_regressors
                if variable.name == self._x.name and variable.wave > regression.lval.wave  # pyright: ignore[reportOperatorIssue]
            ]

            if len(x_future) != 0:
                self._covariances.append(Covariance(y_current, x_future))


class CSModelDefinitionBuilder(_ModelDefinitionBuilder):
    def with_y(self, y: VariableDefinition) -> Self:
        if y.dummy_levels is not None:
            raise ValueError("Cannot have dependent variable be a dummy variable, must be interval scale.")  # noqa: TRY003

        self._y = y

        return self

    def with_x(self, x: VariableDefinition) -> Self:
        self._x = x

        return self

    def with_additional_covariances(
        self,
        *,
        within_dummy_covariance: bool = True,
    ) -> Self:
        self._do_add_dummy_covariances = within_dummy_covariance
        return self

    def build(self, data: pd.DataFrame, *, drop_first_dummy: bool = True) -> str:
        assert_column_type_correct(data)

        available_variables: list[Column] = list(data.columns)  # pyright: ignore[reportAssignmentType]

        self._build_regression(data, available_variables, drop_first_dummy)

        if self._do_PD_check:
            self._check_covariance_matrix_PD(data)

        return self._make_result()

    def _build_regression(
        self,
        data: pd.DataFrame,
        available_variables: list[Column],
        drop_first_dummy: bool,  # noqa: FBT001
    ):
        # Majorly copy-pasta'd from PanelModelDefinitionBuilder._build_regressions

        y = Variable(self._y.name)
        x = self._compile_regressors([self._x], drop_first_dummy, "beta0")

        mediators = self._compile_regressors(self._mediators, drop_first_dummy, "zeta")

        controls = self._compile_regressors(self._controls, drop_first_dummy)
        rvals: list[Variable] = [*x, *mediators, *controls]

        rvals = self._filter_excluded_regressors(rvals)

        if self._do_missing_check:
            skip_regression, rvals = self._filter_missing_rvals(y, rvals, available_variables)

            if skip_regression:
                return

        if self._do_variance_check:
            rvals = self._filter_constant_rvals(y, rvals, data)

        self._regressions.append(Regression(y, rvals, self._include_time_dummy))
        self._add_mediator_regressions(mediators, x, controls)

        if self._do_add_dummy_covariances:
            self._add_dummy_covariances(rvals)

        self._define_ordinals([y], x, mediators, controls)

    def _compile_regressors(
        self,
        regressors: list[VariableDefinition],
        drop_first_dummy: bool,  # noqa: FBT001
        parameter_name: str | None = None,
    ) -> list[Variable]:
        # if wave_y is None, it's a cross-sectional regression, else panel regression.
        result: list[Variable] = []

        for regressor in regressors:
            if regressor.dummy_levels is None:
                regressor_as_variable = Variable(regressor.name)

                if parameter_name is not None:
                    regressor_as_variable = regressor_as_variable.with_named_parameter(
                        f"{parameter_name}_{regressor_as_variable.as_parameter_name()}"
                    )

                result.append(regressor_as_variable)

                continue

            dummy_levels = regressor.dummy_levels

            if drop_first_dummy:
                dropped, *dummy_levels = dummy_levels
                logger.debug(f"Dropped first dummy level '{dropped}' for {regressor.name}")

            for level in dummy_levels:
                regressor_as_variable = Variable(regressor.name, dummy_level=level)

                if parameter_name is not None:
                    regressor_as_variable = regressor_as_variable.with_named_parameter(
                        f"{parameter_name}_{regressor_as_variable.as_parameter_name()}"
                    )

                result.append(regressor_as_variable)

        return result

    def _add_mediator_regressions(
        self,
        mediators: Sequence[Variable],
        x_levels: Sequence[Variable],
        controls: Sequence[Variable],
    ):
        for mediator in mediators:
            current_rvals: list[Variable] = []

            for level in x_levels:
                current_rvals.append(  # noqa: PERF401
                    level.with_named_parameter(f"eta_{mediator.as_parameter_name()}_x_{level.dummy_level}")
                )

            current_rvals.extend(controls)

            self._regressions.append(Regression(mediator.to_unnamed(), current_rvals, self._include_time_dummy))

            self._mediator_pathways.extend(
                [
                    MediatorPathway(
                        mediator,
                        f"zeta_{mediator.as_parameter_name()}",
                        [f"eta_{mediator.as_parameter_name()}_x_{level.dummy_level}"],
                        level.dummy_level,
                    )
                    for level in x_levels
                ]
            )

    def _add_dummy_covariances(self, rvals: list[Variable]):
        """Adds the covariances between dummy levels for the given rvals (lvals must be interval scale).

        Should thus be called once for each regression."""
        # NOTE/TODO: Because this function works on the rvals and not self._controls,
        # it does not include a covariance for the dummy level that is excluded for identification.
        # I'm not 100% that is correct behaviour.

        variables = groupby(rvals, lambda rval: rval.name)

        for name, values in variables:
            dummy_levels = [rval for rval in values if rval.name == name and rval.dummy_level is not None]

            if len(dummy_levels) == 0:
                continue

            for i in range(len(dummy_levels) - 1):
                lval = dummy_levels[i].to_unnamed()
                rvals = [dummy_level.to_unnamed() for dummy_level in dummy_levels[i + 1 :]]

                self._covariances.append(Covariance(lval, rvals))
