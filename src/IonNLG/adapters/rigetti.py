import asyncio
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..models import (
    CircuitData,
    CircuitResult,
    DetailedCircuitData,
    Device,
    Experiment,
    Winrate,
)
from .adapter import Adapter

strategy_mapping = {"4q": ("g14_original", "g14")}


class RigettiAdapter(Adapter):
    async def ingest(self):
        futures = []
        loop = asyncio.get_running_loop()
        for strategy in ("4q", "bell_pair"):
            old_strategy_name, file_prefix = strategy_mapping.get(
                strategy, (strategy, strategy)
            )
            folder = self.data_folder / old_strategy_name

            for backend_folder in folder.glob("ankaa*"):
                future = loop.run_in_executor(
                    None,
                    self.load_blocking,
                    self.data_folder,
                    strategy,
                    backend_folder,
                    file_prefix,
                )
                futures.append(future)

        results = await asyncio.gather(*futures)
        return list(itertools.chain.from_iterable(results))

    def load_blocking(
        self, data_folder: Path, strategy: str, backend_folder: Path, file_prefix: str
    ) -> list[tuple[Experiment, DetailedCircuitData]]:
        csv_path = backend_folder / f"{file_prefix}_raw_counts.csv"
        csv_path2 = backend_folder / f"{file_prefix}_win_rate.csv"

        # Load csv with format
        # va,vb,backend,time,question,dataID,instance,shots,mitigated,qubits_used,wiring,bitstrings...
        df = pd.read_csv(csv_path)

        # Same format except win_rate instead of bitstrings...
        df2 = pd.read_csv(csv_path2)

        # We're going to join both to add the win rate column
        keys = ["va", "vb", "backend", "time", "question", "dataID", "instance"]
        df2.set_index(keys, inplace=True)
        cols_to_use = df2.columns.difference(df.columns)
        df = df.join(df2[cols_to_use], on=keys, how="left")

        # Group by dataID as that separates experimental runs
        final_results = []
        for data_id, gdf in df.groupby("dataID"):
            # gdf has columns va,vb,backend,time,question,instance,shots,mitigated,qubits_used,wiring,bitstrings...
            backend: str = gdf["backend"].iloc[0]

            # Skip rigetti's mock device runs
            if "qvm" in backend:
                continue

            shots: int = gdf["shots"].iloc[0]
            experiment_date = str(gdf["time"].iloc[0])  # mmddyyyy
            date = datetime.strptime(experiment_date, "%m%d%Y")

            # Skip invalid circuit runs
            if date < datetime(2024, 9, 27):
                continue

            # Calculate the win rate
            winrates_overview = gdf.groupby("question")["win_rate"].mean()
            winrate = Winrate.from_circuit_winrates(
                86 / 88, gdf["win_rate"].values, shots
            )

            # Extract the histogram
            histogram_results = DetailedCircuitData(results=[])
            for _, row in gdf.iterrows():
                counts = {}
                va, vb = int(row["va"]), int(row["vb"])
                for possibleBitstring, n in row.items():
                    if possibleBitstring.isdecimal() and not np.isnan(n):
                        a, b = (
                            int(possibleBitstring[-2:], 2),
                            int(possibleBitstring[:2], 2),
                        )
                        counts[(a, b)] = n

                try:
                    histogram_results.results.append(
                        CircuitResult(
                            query=(va, vb),
                            win_rate=row["win_rate"],
                            counts=counts,
                        )
                    )
                except ValidationError as e:
                    pass

            attributes = {
                "shots": shots,
                "dataID": data_id,
                "layout": gdf["qubits_used"].iloc[0],
                "wiring": gdf["wiring"].iloc[0],
                **{f"{k}_win_rate": v for k, v in winrates_overview.items()},
            }

            circuit_data = CircuitData(
                strategy=strategy,
                shots=shots,
                num_circuits=len(gdf),
            )

            experiment = Experiment(
                date=date,
                device=Device(
                    platform="superconducting",
                    provider="rigetti",
                    name=backend,
                ),
                win_rate=winrate,
                circuit_data=circuit_data,
                attributes=attributes,
            )

            final_results.append((experiment, histogram_results))

        return final_results
