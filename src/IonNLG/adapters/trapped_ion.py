import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..models import (
    CircuitData,
    CircuitResult,
    DetailedCircuitData,
    Device,
    Experiment,
    Winrate,
)
from .adapter import Adapter

BLUE_SHOTS = SILVER_SHOTS = 2000
GOLD_SHOTS = 300
ARIA_SHOTS = 20000

VERTEX_WEIGHT = 1
EDGE_WEIGHT = 2


@dataclass
class CircuitMapping:
    map: dict[int, tuple[int, int]]

    @property
    def vertex_indices(self):
        return set(k for k, v in self.map.items() if v[0] == v[1])

    @property
    def edge_indices(self):
        return set(self.map) - self.vertex_indices

    @property
    def weights(self):
        return np.array([VERTEX_WEIGHT] * 14 + [EDGE_WEIGHT] * 37, dtype=float) / 88


def order_winrates(winrates: dict[int, float], mapping: CircuitMapping) -> list[float]:
    win_rates = []
    for idx in sorted(mapping.vertex_indices):
        win_rates.append(winrates[idx])
    for idx in sorted(mapping.edge_indices):
        win_rates.append(winrates[idx])

    return win_rates


def get_circuit_mapping(data_folder: Path):
    pattern = re.compile(r"line (\d+) \((\d+), (\d+)\)")
    circuit_dir = data_folder / "circuits"
    mapping = {}
    for file in circuit_dir.glob("*.txt"):
        if m := pattern.match(file.stem):
            n, va, vb = map(int, m.groups())
            mapping[n] = (va, vb)

    return CircuitMapping(mapping)


def get_blue_data(data_folder: Path, mapping: CircuitMapping):
    shots = BLUE_SHOTS
    file = data_folder / "Blue data.txt"
    data: list[dict[int, float]] = eval(file.read_text())
    count_results = DetailedCircuitData()
    winrates = {}
    for idx, probs in enumerate(data):
        w = sum(probs.get(x, 0) for x in {0, 5, 10, 15})

        if idx in mapping.edge_indices:
            w = 1 - w

        winrates[idx] = w

        counts = {}
        for x_int, p in probs.items():
            n = int(np.round(shots * p))
            ca, cb = x_int % 4, x_int // 4
            counts[(ca, cb)] = n

        count_results.results.append(
            CircuitResult(
                query=tuple(mapping.map[idx]),
                win_rate=w,
                counts=counts,
            )
        )

    experiment = Experiment(
        date=datetime(2024, 10, 9),  # fixme: need time for executions
        device=Device(platform="trapped-ion", provider="duke", name="blue"),
        win_rate=Winrate.from_circuit_winrates(
            86 / 88, order_winrates(winrates, mapping), shots, weights=mapping.weights
        ),
        circuit_data=CircuitData(
            strategy="bell_pair",
            shots=shots,
            num_circuits=len(winrates),
        ),
        attributes={
            "vertex_win_rate": np.mean(
                [winrates[x] for x in mapping.vertex_indices]
            ).item(),
            "edge_win_rate": np.mean(
                [winrates[x] for x in mapping.edge_indices]
            ).item(),
        },
    )

    return experiment, count_results


def get_ionq_data(data_folder: Path, mapping: CircuitMapping):
    shots = ARIA_SHOTS
    file = data_folder / "ionq_winrates.json"
    data = json.loads(file.read_text())
    winrates = {int(k): v for k, v in data.items()}

    experiment = Experiment(
        date=datetime(2024, 11, 4),  # fixme: need time for executions
        device=Device(platform="trapped-ion", provider="ionq", name="aria"),
        win_rate=Winrate.from_circuit_winrates(
            86 / 88, order_winrates(winrates, mapping), shots, weights=mapping.weights
        ),
        circuit_data=CircuitData(
            strategy="bell_pair", shots=shots, num_circuits=len(winrates)
        ),
        attributes={
            "vertex_win_rate": np.mean(
                [winrates[x] for x in mapping.vertex_indices]
            ).item(),
            "edge_win_rate": np.mean(
                [winrates[x] for x in mapping.edge_indices]
            ).item(),
        },
    )

    # Make detailed circuit data with None counts since we don't have the raw histogram data
    # for IonQ
    detailed_circuit_data = DetailedCircuitData(
        results=[
            CircuitResult(
                query=mapping.map[idx],
                win_rate=winrates[idx],
                counts=None,
            )
            for idx in sorted(mapping.map)
        ]
    )

    return experiment, detailed_circuit_data


def get_gold_data(data_folder: Path, mapping: CircuitMapping):
    shots = GOLD_SHOTS
    file = data_folder / "Gold data.json"
    data: dict[str, dict[str, float]] = json.loads(file.read_text())
    winrates = {}
    count_results = DetailedCircuitData(results=[])
    for idx_str, probs in data.items():
        idx = int(idx_str)

        counts = {}
        for bitstring, p in probs.items():
            n = int(np.round(p * shots))
            ca, cb = int(bitstring[-2:], 2), int(bitstring[:2], 2)
            counts[(ca, cb)] = n

        probs = {int(k, 2): v for k, v in probs.items()}
        w = sum(probs.get(x, 0) for x in {0, 5, 10, 15})

        if idx in mapping.edge_indices:
            w = 1 - w

        winrates[idx] = w

        count_results.results.append(
            CircuitResult(
                query=tuple(mapping.map[idx]),
                win_rate=w,
                counts=counts,
            )
        )

    experiment = Experiment(
        date=datetime(2024, 11, 18),  # fixme: need time for executions
        device=Device(platform="trapped-ion", provider="duke", name="gold"),
        win_rate=Winrate.from_circuit_winrates(
            86 / 88, order_winrates(winrates, mapping), shots, weights=mapping.weights
        ),
        circuit_data=CircuitData(
            strategy="bell_pair",
            shots=shots,
            num_circuits=len(winrates),
        ),
        attributes={
            "vertex_win_rate": np.mean(
                [winrates[x] for x in mapping.vertex_indices]
            ).item(),
            "edge_win_rate": np.mean(
                [winrates[x] for x in mapping.edge_indices]
            ).item(),
        },
    )

    return experiment, count_results


def get_silver_data(data_folder: Path, mapping: CircuitMapping):
    file = data_folder / "silver_unmitigated.csv"

    df = pd.read_csv(file)
    shots = df["shots"].iloc[0]
    df["question"] = np.where(df["va"] == df["vb"], "vertex", "edge")
    winrates_overview = df.groupby("question").win_rate.mean()
    weights = np.where(df["va"] == df["vb"], VERTEX_WEIGHT, EDGE_WEIGHT) / 88.0
    winrate = Winrate.from_circuit_winrates(
        86 / 88, df["win_rate"].values, shots, weights=weights
    )

    experiment = Experiment(
        date=datetime(2024, 7, 18),
        device=Device(platform="trapped-ion", provider="umd", name="silver"),
        win_rate=winrate,
        circuit_data=CircuitData(
            strategy="bell_pair",
            shots=shots,
            num_circuits=len(df),
        ),
        attributes={f"{k}_win_rate": v for k, v in winrates_overview.items()},
    )

    # Make the detailed circuit data
    detailed_circuit_data = DetailedCircuitData(
        results=[
            CircuitResult(
                query=(int(row["va"]), int(row["vb"])),
                win_rate=row["win_rate"],
                counts=None,
            )
            for _, row in df.iterrows()
        ]
    )

    return experiment, detailed_circuit_data


class TrappedIonAdapter(Adapter):
    async def ingest(self) -> list[tuple[Experiment, DetailedCircuitData]]:
        mapping = get_circuit_mapping(self.data_folder)
        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(None, func, self.data_folder, mapping)
            for func in (get_silver_data, get_gold_data, get_blue_data, get_ionq_data)
        ]

        results = await asyncio.gather(*futures)
        return results
