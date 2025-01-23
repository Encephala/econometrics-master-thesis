from typing import Self
import warnings

import pandas as pd


class ModelBuilder:
    y: str
    y_ordinal: bool

    x: str
    x_ordinal: bool

    w: list[str]  # TODO
    lag_structure: list[int]  # TODO

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
        raise NotImplementedError

        self.lag_structure = lag_structure
        return self

    def build(self, data: pd.DataFrame) -> str:
        result = []

        columns = data.columns

        y_columns = [col for col in columns if col.startswith(self.y)]
        y_start = min((int(column[column.find("_") + 1 :]) for column in y_columns))
        y_end = max((int(column[column.find("_") + 1 :]) for column in y_columns))

        x_columns = [col for col in columns if col.startswith(self.x)]
        x_start = min((int(column[column.find("_") + 1 :]) for column in x_columns))
        x_end = max((int(column[column.find("_") + 1 :]) for column in x_columns))

        # TODO?: Yield some information on the breadth of the overlap?
        overlap_start = max(x_start, y_start)
        overlap_end = min(x_end, y_end)

        # For now, assumes lag structure: (1)
        # Hence, build a regression for each year after the first
        for year in range(overlap_start + 1, overlap_end):
            y_name = f"{self.y}_{year}"
            x_name = f"{self.x}_{year}"

            if y_name not in columns or x_name not in columns:
                warnings.warn(f"For {year=}, either {y_name} or {x_name} was not found", stacklevel=2)
                continue

            regression = f"{y_name} ~ {x_name}\n"
            result.append(regression)

            if self.y_ordinal is not None:
                result.append(f"DEFINE(ordinal) {y_name}\n")

            if self.x_ordinal is not None:
                result.append(f"DEFINE(ordinal) {x_name}\n")

        # TODO: implement some stuff to make it less likely that I forget to properly order an ordinal column

        if len(result) == 0:
            warnings.warn("Result empty, no model defined", stacklevel=2)

        return "\n".join(result)


if __name__ == "__main__":
    ModelBuilder().build(pd.DataFrame())
