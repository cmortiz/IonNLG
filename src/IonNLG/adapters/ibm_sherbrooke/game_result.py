import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rustworkx as rx
from qiskit_ibm_runtime.ibm_backend import BackendProperties


@dataclass
class GameResult:
    """Class for holding the results of a batch of game circuits, including
    noise characterization and calibration data"""

    job_id: str
    """IBM job id for the game and noise circuits"""

    backend: str
    """Name of the backend used"""

    strategy: str
    """The particular game strategy that was run"""

    win_rate: rx.PyDiGraph = None
    """Graph of the game"""

    mitigated_win_rate: rx.PyDiGraph = None
    """Graph of the game after readout error mitigation"""

    spam_matrices: Dict[str, np.ndarray] = field(default_factory=dict)
    """SPAM matrix in x and z basis"""

    counts: Dict[Tuple[int, int], Dict[str, int]] = field(default_factory=dict)
    """Histogram of bitstrings for each question"""

    mirror_counts: Dict[int, int] = field(default_factory=dict)
    """Counts for the mirror state preparation circuit"""

    calibration_data: BackendProperties = None
    """Most recent calibration data for the backend at the time the circuits were executed"""

    # Todo: Extend this to include bell pair prep circuits.

    @property
    def questions(self):
        return self.counts.keys()

    def get_adjacency_matrix(self, G: rx.PyDiGraph = None) -> np.ndarray:
        """Returns the win rate as an adjacency matrix.

        By providing other graphs, this can be used to e.g., get the win rate
        after applying readout error mitigation.

        Args:
            G: The graph to use. If not provided, the win rate graph will be used.
        """

        G = G or self.win_rate
        A = rx.adjacency_matrix(G, weight_fn=lambda x: x, null_value=np.nan)
        for node_idx in G.node_indices():
            A[node_idx, node_idx] = G[node_idx]

        return A

    @property
    def mirror_fidelity(self) -> float:
        shots = sum(self.mirror_counts.values())
        mirror_fidelity = self.mirror_counts[0] / shots
        return mirror_fidelity

    def crosstalk(self, basis="z"):
        """Estimate the crosstalk from the spam matrix [1].

        First we construct an approximate spam matrix by marginalizing over each qubit
        to construct its own 2x2 spam matrix. Then we tensor those together to obtain
        S'. Under no crosstalk, S' = S. Therefore, we return the distance ||S - S'||.

        [1] Hamilton, Kathleen E., et al. "Scalable quantum processor noise characterization."
            2020 IEEE International Conference on Quantum Computing and Engineering (QCE).
            IEEE, 2020.
        """
        S = self.spam_matrices[basis.lower()]
        states = S.shape[0]
        qubits = int(np.log2(states))

        small_matrices = []
        # Marginalize over each qubit. This produces a matrix
        # S'_i = [p_00 p_01; p_10 p_11] = [p_00 1-p_11; 1-p_00 p_11]
        for qubit in range(qubits):
            si = np.zeros((2, 2))
            for prepared in (0, 1):
                if prepared == 0:
                    # Set q=0
                    idx = np.unique(np.arange(states) & ~(1 << qubit))
                else:
                    # Set q=1
                    idx = np.unique(np.arange(states) | (1 << qubit))

                si[prepared, prepared] = S[idx, idx].mean()
                si[1 - prepared, prepared] = 1 - si[prepared, prepared]

            small_matrices.append(si)

        # Tensor the matrices back together to arrive at S'
        S_prime = small_matrices[0]
        if qubits > 1:
            for si in small_matrices[1:]:
                S_prime = np.kron(S_prime, si)

        # Compute the distance from the original spam matrix
        return np.linalg.norm(S - S_prime, ord="fro")

    def to_record(self, *, include_qubit_metrics=False) -> dict:
        A = self.get_adjacency_matrix()
        vertex_win_rate = A.trace() / A.shape[0]

        # Delete the diagonal
        i = np.arange(A.shape[0])
        A[i, i] = np.nan

        # Now take the mean of everything else
        edge_win_rate = np.nanmean(A)

        # Obtain average fidelity of preparing a state
        spam_data = {}
        for basis, matrix in self.spam_matrices.items():
            mean_fidelity = matrix.trace() / matrix.shape[0]
            spam_data[f"{basis}_spam_fidelity"] = mean_fidelity

        # Add qubit metrics if requested, which involves loading the layout of the
        # circuits from qiskit cloud, then looking through the calibration data.
        # Todo: Add this
        if include_qubit_metrics:
            # I actually don't know what properties in the qiskit Job object
            # allow us to retrieve the circuit. It stands to reason that they
            # stored it since if you visit the job page, you can see it display
            # the circuit.
            raise NotImplementedError()

        return {
            "job_id": self.job_id,
            "backend": self.backend,
            "strategy": self.strategy,
            "vertex_win_rate": vertex_win_rate,
            "edge_win_rate": edge_win_rate,
            **spam_data,
            "mirror_fidelity": self.mirror_fidelity,
            "crosstalk": self.crosstalk(),
        }

    @classmethod
    def load_from_folder(cls, job_folder: str | Path):
        """Loads the results from a folder.

        Note we assume the folder is named /path/to/data/<experiment_id>/raw/<job_id>,
        so we extract the job id from the path
        """

        # Get job id
        job_folder = Path(job_folder)
        job_id = job_folder.stem

        # Get the backend from the experiment folder
        metadata_file = job_folder.parent.parent.resolve() / "metadata.json"
        metadata = json.loads(metadata_file.read_text("utf-8"))
        backend = metadata["backend"]

        # Load the game data. Within the folder there should be a game/<strategy> folder
        strategy = next((job_folder / "game").iterdir()).stem
        game_folder = job_folder / "game" / strategy
        win_rate, counts_per_question = cls._load_game_win_rate(game_folder)

        # Load the spam matrices
        spam_matrices = {}
        for basis in ("x", "z"):
            spam_matrices[basis] = cls._load_spam_circuits(job_folder, basis)

        mitigated_win_rate, _ = cls._load_game_win_rate(game_folder, spam_matrices["z"])

        # Fetch more noise results
        mirror_counts = cls._get_mirror_counts(job_folder, strategy)
        calibration_data = cls._import_calibration_data(job_folder)

        return cls(
            job_id,
            backend,
            strategy,
            win_rate,
            mitigated_win_rate,
            spam_matrices,
            counts_per_question,
            mirror_counts,
            calibration_data,
        )

    @staticmethod
    def _load_spam_circuits(job_folder: str | Path, basis: str = "z"):
        basis = basis.upper()
        assert basis in ("X", "Y", "Z"), f"Unrecognized basis: {basis}"

        job_folder = Path(job_folder)
        spam_folder = job_folder / "noise" / "spam_matrix"
        prefix = f"{basis}basis_SPAM*"

        results = {}
        for circuit_folder in spam_folder.glob(prefix):
            # Check that we have a valid folder
            if not circuit_folder.is_dir():
                continue

            # Parse the prepared state from binary
            prepared_state = circuit_folder.name.split("_")[-1]
            prepared_state = int(prepared_state, 2)

            # Load the counts, mapping bitstrings to ints
            counts_file = circuit_folder / "counts.json"
            counts = json.loads(counts_file.read_text("utf-8"))
            counts = {int(k, 2): v for k, v in counts.items()}
            results[prepared_state] = counts

        # Now from our nested dict, we can construct a numpy array.
        # Here we assume the prepared states were 0, ..., max(results.keys())
        n = max(results.keys()) + 1
        spam_matrix = np.zeros((n, n))

        for prepared_state, counts in results.items():
            for outcome, count in counts.items():
                spam_matrix[outcome, prepared_state] = count

        # Normalize by the shot count
        shots = spam_matrix.sum(axis=0)
        assert (
            shots == shots[0]
        ).all(), "Not all states were prepared with the same shots"
        spam_matrix = spam_matrix / shots[0]

        return spam_matrix

    @staticmethod
    def _load_game_win_rate(
        game_folder: str | Path,
        spam_matrix: np.ndarray = None,
    ) -> rx.PyDiGraph:
        """Loads the game win rate as a graph

        Args:
            game_folder: The folder containing the game circuit results

            spam_matrix: A SPAM matrix in the Z basis. If provided, this will be used
                to perform readout-error mitigation
        """

        # Scan the folder and get the number of vertices
        game_folder = Path(game_folder)
        vertices = set()
        for circuit_folder in game_folder.iterdir():
            if not circuit_folder.is_dir():
                continue

            # Load the question (va, vb)
            *_, va, vb = circuit_folder.name.split("_")
            va, vb = int(va), int(vb)

            vertices.add(va)
            vertices.add(vb)

        # Construct our graph with empty nodes. By doing it up front, this ensures
        # each vertex matches the internal node id.
        G = rx.PyDiGraph()
        G.add_nodes_from(range(min(vertices), max(vertices) + 1))

        counts_per_question = {}
        for circuit_folder in game_folder.iterdir():
            if not circuit_folder.is_dir():
                continue

            # Load the question (va, vb)
            *_, va, vb = circuit_folder.name.split("_")
            va, vb = int(va), int(vb)
            is_vertex_question = va == vb

            # Get the counts and calculate the win rate.
            # Todo: Calculate variance? That was one of the questions
            # at QCUF.
            counts = json.loads((circuit_folder / "counts.json").read_text("utf-8"))
            counts_per_question[(va, vb)] = counts
            probs = _counts_to_probability(counts)

            # Perform readout-error mitigation if necessary
            if spam_matrix is not None:
                probs = np.linalg.inv(spam_matrix) @ probs

            # Number of possible colors is the square root of dim(probs). Calculate
            # the integer indices which correspond to matching colors. For a game
            # with 4 colors, this would be 0, 5, 10, 15.
            dim = probs.shape[0]
            colors = int(np.sqrt(dim))
            bits = int(np.log2(dim))
            c = np.arange(colors)
            idx = c + c * 2 ** (bits // 2)
            vertex_win_rate = probs[idx].sum().item()
            edge_win_rate = 1 - vertex_win_rate

            if is_vertex_question:
                G[va] = vertex_win_rate
            else:
                G.add_edge(va, vb, edge_win_rate)

        return G, counts_per_question

    @staticmethod
    def _get_mirror_counts(job_folder: str | Path, strategy: str):
        job_folder = Path(job_folder)
        counts_file = job_folder / "noise" / "mirror" / strategy / "counts.json"
        counts = json.loads(counts_file.read_text("utf-8"))
        counts = {int(k, 2): v for k, v in counts.items()}

        return counts

    @staticmethod
    def _import_calibration_data(job_folder: str | Path) -> BackendProperties:
        # Import the calibration data
        job_folder = Path(job_folder)
        calibration_data = json.loads(
            (job_folder / "calibration_data.json").read_text("utf-8")
        )

        # Replace date strings with datetime objects (which was necessary to
        # serialize in the first place)
        def replace_date_strings(d):
            if isinstance(d, list):
                for item in d:
                    replace_date_strings(item)
            elif isinstance(d, dict):
                for key, value in d.items():
                    if "date" in key and isinstance(value, str):
                        d[key] = datetime.fromisoformat(value.removesuffix("Z"))
                    else:
                        replace_date_strings(value)

        replace_date_strings(calibration_data)
        return BackendProperties.from_dict(calibration_data)


def _counts_to_probability(counts: Dict[str, int]) -> np.ndarray:
    max_bitstring = int(max(counts.keys()), 2)
    n = int(np.ceil(np.log2(max_bitstring + 1)))
    probs = np.zeros(2**n)

    for bitstring, count in counts.items():
        outcome = int(bitstring, 2)
        probs[outcome] = count

    probs /= probs.sum()

    return probs
