import ast
from datetime import datetime
from typing import Annotated, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
)

from . import uncertainty


def validate_tuple_keys(v: dict[str, Any]) -> dict[tuple[int, ...], Any]:
    """
    Validator function to convert string representations of tuples back to actual tuples.
    """
    if not isinstance(v, dict):
        # This handles cases where the input is not a dict (e.g., None, a string, etc.)
        # Pydantic's core validation will then handle if it's not the right overall type.
        return v  # or raise TypeError("Input must be a dictionary") if stricter validation is needed here

    new_dict = {}
    for key_str, value in v.items():
        if isinstance(key_str, tuple):
            new_dict[key_str] = value
            continue

        try:
            # Safely evaluate the string as a Python literal (e.g., tuple, int, float, list, dict, string, bool, None)
            evaluated_key = ast.literal_eval(key_str)
            # Ensure the evaluated key is indeed a tuple and matches the expected type
            if not isinstance(evaluated_key, tuple):
                raise ValueError(f"Key '{key_str}' is not a valid tuple string.")
            # Optionally, you can add more specific validation for the tuple's contents
            # For example, to ensure all elements are integers:
            if not all(isinstance(item, int) for item in evaluated_key):
                raise ValueError(
                    f"Tuple key '{key_str}' contains non-integer elements."
                )

            new_dict[evaluated_key] = value
        except (ValueError, SyntaxError) as e:
            # If ast.literal_eval fails, it means the string is not a valid literal
            raise ValueError(
                f"Invalid tuple key string received: '{key_str}'. Error: {e}"
            ) from e
    return new_dict


Query = tuple[int, ...]
Histogram = Annotated[dict[Query, Any], BeforeValidator(validate_tuple_keys)]


class Device(BaseModel):
    platform: str
    provider: str
    name: str


class CircuitData(BaseModel):
    strategy: str
    shots: int
    num_circuits: int


class DetailedCircuitData(BaseModel):
    results: list["CircuitResult"] = Field(default_factory=list)

    def to_df(self) -> pd.DataFrame:
        records = []
        for circuit_result in self.results:
            query = {f"x{i}": v for i, v in enumerate(circuit_result.query)}
            d = query | {"win_rate": circuit_result.win_rate}

            if circuit_result.counts is not None:
                for outcome, count in circuit_result.counts.items():
                    record = d.copy()
                    a = {f"a{i}": v for i, v in enumerate(outcome)}
                    record = record | a | {"count": count}
                    records.append(record)
            else:
                records.append(d)

        return pd.DataFrame.from_records(records)


class CircuitResult(BaseModel):
    query: Query
    """Query for the players"""

    win_rate: float
    """Success rate of the circuit"""

    counts: Histogram | None = None
    """Raw histogram of the responses, if available"""


class Winrate(BaseModel):
    value: float
    ci95: float
    p_value: float
    var: float

    @classmethod
    def from_circuit_winrates(
        cls,
        classical_game_value: float,
        winrates: ArrayLike,
        shots: int,
        weights: np.ndarray | None = None,
    ):
        var = uncertainty.calculate_var(winrates, shots, weights)
        return cls(
            value=uncertainty.calculate_mean(winrates, weights),
            ci95=uncertainty.calculate_ci(
                winrates, shots, d=0.05, weights=weights, var=var
            ),
            p_value=uncertainty.calculate_p_value(
                winrates, shots, classical_game_value, weights=weights, var=var
            ),
            var=var,  # type: ignore
        )

    def to_str(self, decimals=1):
        # Displays a win rate with a confidence interval, e.g. 98.67(14),
        # where the number(s) in parentheses is the error on the last digit(s)
        winrate, err = self.value, self.ci95
        winrate *= 100
        err *= 100 * 10**decimals

        # Format win rate with decimals
        winrate = np.floor(winrate * 10**decimals) / 10**decimals
        winrate = f"{winrate:.{decimals}f}"

        # Format the error to only keep nonzero digits
        err = int(np.ceil(err))
        return f"{winrate}({err:d})"


class Experiment(BaseModel):
    date: datetime
    """Time of the experiment"""

    device: Device
    """Quantum device the experiment was executed on"""

    win_rate: Winrate

    circuit_data: CircuitData

    attributes: dict[str, bool | int | float | str] = Field(default_factory=dict)
    """Any extra data attributes"""

    def to_summary_df(self) -> pd.DataFrame:
        data = {
            "date": self.date,
            "device_platform": self.device.platform,
            "device_provider": self.device.provider,
            "device_name": self.device.name,
            "win_rate_value": self.win_rate.value,
            "win_rate_ci95": self.win_rate.ci95,
            "win_rate_p_value": self.win_rate.p_value,
            "win_rate_var": self.win_rate.var,
            "win_rate_str": self.win_rate.to_str(),
            **{f"attr_{k}": v for k, v in self.attributes.items()},
        }
        return pd.DataFrame([data])
