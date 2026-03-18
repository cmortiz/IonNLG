from datetime import datetime

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


class Ibm2023Adapter(Adapter):
    async def ingest(self) -> list[tuple[Experiment, DetailedCircuitData]]:
        final_results = []
        processed_csv = self.data_folder / "ibm_processed.csv"
        raw_csv = self.data_folder / "ibm_results.csv"

        df = pd.read_csv(processed_csv)
        raw_df = pd.read_csv(raw_csv)

        for job_id, gdf in df.groupby("job"):
            first_row = gdf.iloc[0]
            shots = first_row.shots
            device = Device(
                platform="superconducting",
                provider="ibm",
                name=first_row.backend.split("_", 1)[-1],
            )
            date = datetime.fromisoformat(first_row.time)

            winrate = Winrate.from_circuit_winrates(86 / 88, gdf.q_winrate, shots)
            winrates_by_type = gdf.groupby("qtype").q_winrate.mean()

            # Obtain the counts from the raw csv that has the columns ca, cb, n
            count_results = DetailedCircuitData(results=[])
            for _, row in gdf.iterrows():
                va, vb = row["va"], row["vb"]
                raw_gdf = raw_df.loc[(raw_df["va"] == va) & (raw_df["vb"] == vb)]
                counts = {}
                for ca, cb, n in zip(raw_gdf["ca"], raw_gdf["cb"], raw_gdf["n"]):
                    counts[(ca, cb)] = n

                count_results.results.append(
                    CircuitResult(
                        query=(int(va), int(vb)),
                        win_rate=row["q_winrate"],
                        counts=counts,
                    )
                )

            circuit_data = CircuitData(
                strategy="4q",
                shots=shots,
                num_circuits=len(gdf),
            )
            experiment = Experiment(
                date=date,
                device=device,
                win_rate=winrate,
                circuit_data=circuit_data,
                attributes={
                    "job_id": str(job_id),
                    "vertex_win_rate": winrates_by_type["Vertex"],
                    "edge_win_rate": winrates_by_type["Edge"],
                },
            )

            final_results.append((experiment, count_results))

        return final_results
