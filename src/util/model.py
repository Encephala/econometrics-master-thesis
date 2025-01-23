from typing import Self
import warnings

import pandas as pd


class ModelBuilder:
    y: str
    x: str
    w: list[str]  # TODO
    lag_structure: list[int]  # TODO

    def with_y(self, y: str) -> Self:
        self.y = y
        return self

    def with_x(self, x: str) -> Self:
        self.x = x
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
        result = ""

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
            regression = f"{self.y}_{year} ~ {self.x}_{year - 1}\n"

            result += regression

        if result == "":
            warnings.warn("Result empty, no model defined", stacklevel=2)

        return result


if __name__ == "__main__":
    ModelBuilder().build(pd.DataFrame())
