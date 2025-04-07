import pandas as pd
from IPython.display import display
import semopy


def print_results(model: semopy.Model):
    # https://stackoverflow.com/a/20937592/5410751
    results = model.inspect().sort_values(["op", "rval"])  # pyright: ignore noqa: PGH003

    with pd.option_context("display.float_format", "{:.4f}".format, "display.max_rows", None):
        display(results)
