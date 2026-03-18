import asyncio
from pathlib import Path

import pandas as pd

from .adapters import (
    Adapter,
    Ibm2023Adapter,
    IbmSherbrookeAdapter,
    RigettiAdapter,
    TrappedIonAdapter,
)
from .models import Experiment

data_folder = Path("data")
raw_data_folder = data_folder / "raw_data"
raw_data_folder.mkdir(exist_ok=True, parents=True)
processed_data_folder = data_folder / "processed_data"
processed_data_folder.mkdir(exist_ok=True, parents=True)


async def main():
    experiments: list[Experiment] = []

    # Import all the superconducting data and save only the high-level summary results
    # since those data are not from this experiment
    superconducting_adapters: list[Adapter] = [
        Ibm2023Adapter(raw_data_folder / "superconducting" / "ibm_2023"),
        IbmSherbrookeAdapter(raw_data_folder / "superconducting" / "ibm_2024"),
        RigettiAdapter(raw_data_folder / "superconducting" / "rigetti_2024"),
    ]

    futures = [adapter.ingest() for adapter in superconducting_adapters]
    for future in asyncio.as_completed(futures):
        result = await future
        for experiment, _ in result:
            experiments.append(experiment)

    # Import the trapped ion data but also save the detailed results since those data
    # *are* from this experiment
    trapped_ion_adapter = TrappedIonAdapter(raw_data_folder / "trapped_ion")
    result = await trapped_ion_adapter.ingest()
    for experiment, detailed_circuit_data in result:
        experiments.append(experiment)

        file = (
            processed_data_folder
            / "trapped_ion"
            / f"{experiment.device.provider}_{experiment.device.name}.csv"
        )
        file.parent.mkdir(exist_ok=True, parents=True)
        df = detailed_circuit_data.to_df()
        df.to_csv(file, index=False)

    summary_df = pd.concat(
        [experiment.to_summary_df() for experiment in experiments], ignore_index=True
    )
    summary_file = processed_data_folder / "summary.csv"
    summary_df.to_csv(summary_file, index=False)


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
