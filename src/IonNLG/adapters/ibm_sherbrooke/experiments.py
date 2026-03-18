import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.runtime_job_v2 import RuntimeJobV2


def get_experiments(data_dir: Path, service: QiskitRuntimeService, *, real_only=False):
    experiments = []
    for experiment_folder in data_dir.iterdir():
        if not experiment_folder.is_dir():
            continue

        metadata = json.loads((experiment_folder / "metadata.json").read_text("utf-8"))

        # Convert the submitted time (which is ns since epoch) to
        # datetime object
        submitted = datetime.fromtimestamp(metadata["submitted"] / 1e9)

        # Obtain all the strategies run in this Batch
        strategies = set()
        for circuit_name in metadata["circuits"]:
            if circuit_name.startswith("game"):
                strategies.add(circuit_name.split(".")[1])

        strategies = list(strategies)

        # Determine the job status
        if "fake" in metadata["backend"]:
            status = "DONE"

            # Skip if we're ignoring simulators
            if real_only:
                continue
        else:
            # Caching: store a terminal status in a status.txt file
            # in the experiment directory
            statuses = set()
            cache_file = experiment_folder / "status.txt"
            if cache_file.exists():
                statuses.add(cache_file.read_text("utf-8").strip())

            else:
                for job_id in metadata["job_id"]:
                    job = service.job(job_id)
                    statuses.add(job.status())

            if any(s not in RuntimeJobV2.JOB_FINAL_STATES for s in statuses):
                status = "PENDING"
            else:
                status = statuses.pop()

                # Store in the cache if it's a terminal state
                if not cache_file.exists():
                    cache_file.write_text(status)

        experiments.append(
            {
                "id": metadata["experiment_id"],
                "backend": metadata["backend"],
                "status": status,
                "submitted": submitted,
                "downloaded": (experiment_folder / "raw.zip").exists(),
                "jobs": metadata["job_id"],
                "strategies": strategies,
            }
        )

    df = pd.DataFrame.from_records(experiments)
    df.sort_values("submitted", inplace=True, ignore_index=True)
    return df
