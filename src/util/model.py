from typing import Self
import warnings

import pandas as pd


class ModelDefinitionBuilder:
    y: str
    y_ordinal: bool

    x: str
    x_ordinal: bool

    w: list[str] | None = None  # TODO
    lag_structure: list[int] | None = None

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

    def build(self, data: pd.DataFrame) -> str:
        result = []

        columns = data.columns

        y_columns = [col for col in columns if col.startswith(self.y)]
        y_start = min(int(column[column.find("_") + 1 :]) for column in y_columns)
        y_end = max(int(column[column.find("_") + 1 :]) for column in y_columns)

        x_columns = [col for col in columns if col.startswith(self.x)]
        x_start = min(int(column[column.find("_") + 1 :]) for column in x_columns)
        x_end = max(int(column[column.find("_") + 1 :]) for column in x_columns)

        # Build a regression starting max_lag years after the overlap
        lag_structure = self.lag_structure if self.lag_structure is not None else [1]

        # TODO?: Yield some information on the breadth of the overlap?
        # TODO?: When using FIML/handling missing data, perhaps the `max` and `min` in these two statements should swap,
        # but then we have to also add those columns to the df to prevent index errors

        # if x goes far enough back, start when y starts
        # else, start as soon as we can
        first_year_y = max(y_start, x_start + max(lag_structure))
        # if x goes far enough forward, stop when x_stops
        # else, stop when y stops
        last_year_y = min(x_end + min(lag_structure), y_end)

        for year_y in range(first_year_y, last_year_y):
            y_name = f"{self.y}_{year_y}"
            x_names = [f"{self.x}_{year_y - lag}" for lag in lag_structure]

            if y_name not in columns or any(x_name not in columns for x_name in x_names):
                warnings.warn(f"For {year_y=}, either {y_name} or one of {x_names} was not found", stacklevel=2)
                continue

            regression = f"{y_name} ~ {' + '.join(x_names)}"
            result.append(regression)

            if self.y_ordinal:
                result.append(f"DEFINE(ordinal) {y_name}")

            if self.x_ordinal:
                result.append(f"DEFINE(ordinal) {' '.join(x_names)}")

        # TODO?: implement some stuff to make it less likely that I forget to properly order an ordinal column

        if len(result) == 0:
            warnings.warn("Result empty, no model defined", stacklevel=2)

        return "\n".join(result)


if __name__ == "__main__":
    ModelDefinitionBuilder().build(pd.DataFrame())
