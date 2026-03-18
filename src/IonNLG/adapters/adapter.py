from abc import ABC, abstractmethod
from pathlib import Path

from ..models import DetailedCircuitData, Experiment


class Adapter(ABC):
    """Interface for a data adapter that imports data into the database"""

    def __init__(self, data_folder: Path):
        self.data_folder = data_folder

    @abstractmethod
    async def ingest(self) -> list[tuple[Experiment, DetailedCircuitData]]:
        """Ingests the data, storing it internally"""
