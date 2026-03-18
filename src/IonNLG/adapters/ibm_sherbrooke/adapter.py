import asyncio
import itertools
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ...models import (
    CircuitData,
    CircuitResult,
    DetailedCircuitData,
    Device,
    Experiment,
    Winrate,
)
from ..adapter import Adapter
from .experiments import get_experiments
from .game_result import GameResult

# Have to manually look this up now since ibm migrated to their new cloud platform
JOB_DATES = {
    "cvvdwwvy3grg008nhj60": "2024-09-27T12:33:56.145000-04:00",
    "cvw4q2w7cb40008e8qag": "2024-09-28T14:31:39.287000-04:00",
    "cvwc3gp9z730008396v0": "2024-09-28T22:56:03.099000-04:00",
    "cvwr0eh5sat0008j19a0": "2024-09-29T12:28:42.446000-04:00",
    "cvxc3y75sat0008j2pa0": "2024-09-30T11:21:29.282000-04:00",
    "cvy2f1cwqn10008w5evg": "2024-10-01T12:47:01.332000-04:00",
    "cvy2j50smnpg0082p62g": "2024-10-01T12:53:40.784000-04:00",
    "cvy53jyfd8tg008xdqv0": "2024-10-01T15:47:24.044000-04:00",
    "cvy59p6smnpg0082pfg0": "2024-10-01T16:00:24.905000-04:00",
}


class IbmSherbrookeAdapter(Adapter):
    async def ingest(self):
        experiments = get_experiments(self.data_folder, None, real_only=True)

        # Filter to only keep complete and downloaded ones. Also remove all runs that had invalid circuits
        mask = (experiments.status == "DONE") & experiments.downloaded
        submitted_times = pd.to_datetime(experiments["submitted"])
        mask &= submitted_times >= pd.Timestamp(datetime(2024, 9, 27))
        experiments = experiments.loc[mask]

        loop = asyncio.get_running_loop()
        futures = []
        for _, experiment in experiments.iterrows():
            future = loop.run_in_executor(
                None, self._load_experiment_blocking, experiment, self.data_folder
            )
            futures.append(future)

        results: list[
            list[tuple[Experiment, DetailedCircuitData]]
        ] = await asyncio.gather(*futures)
        return list(itertools.chain.from_iterable(results))

    def _load_experiment_blocking(
        self, experiment: pd.Series, data_dir: Path
    ) -> list[tuple[Experiment, DetailedCircuitData]]:
        zipfile = data_dir / experiment["id"] / "raw.zip"
        tmpdir = data_dir / experiment["id"] / "raw"
        shutil.unpack_archive(zipfile, tmpdir)
        return_results: list[tuple[Experiment, DetailedCircuitData]] = []

        for job_folder in tmpdir.iterdir():
            result = GameResult.load_from_folder(job_folder)
            record = result.to_record()
            attributes = {
                k: record[k] for k in {"job_id", "vertex_win_rate", "edge_win_rate"}
            }

            winrate_A = result.get_adjacency_matrix()
            winrates = winrate_A.flat
            winrates = winrates[~np.isnan(winrates)]
            unique_shots = set(map(_counts_to_shots, result.counts.values()))

            if len(unique_shots) != 1:
                raise ValueError(
                    f"Obtained multiple distinct shot counts for experiment {experiment['id']}:",
                    unique_shots,
                )

            shots = unique_shots.pop()
            data = Experiment(
                date=JOB_DATES.get(result.job_id),  # pyright: ignore[reportArgumentType]
                device=Device(
                    platform="superconducting",
                    provider="ibm",
                    name=experiment.backend.split("_", 1)[-1],
                ),
                win_rate=Winrate.from_circuit_winrates(86 / 88, winrates, shots),
                circuit_data=CircuitData(
                    strategy=record["strategy"],
                    shots=shots,
                    num_circuits=len(winrates),
                ),
                attributes=attributes,
            )

            circuit_result = DetailedCircuitData(results=[])
            for x, hist in result.counts.items():
                new_hist = {}
                for bitstring, n in hist.items():
                    a = int(bitstring[-2:], 2)
                    b = int(bitstring[:2], 2)
                    new_hist[(a, b)] = n

                circuit_result.results.append(
                    CircuitResult(
                        query=x,
                        win_rate=winrate_A[x],
                        counts=new_hist,
                    )
                )

            return_results.append((data, circuit_result))

        # Cleanup the extracted data
        shutil.rmtree(tmpdir)
        return return_results


def _counts_to_shots(counts: dict[str, int]):
    return sum(counts.values())
