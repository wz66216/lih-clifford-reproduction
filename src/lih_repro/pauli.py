from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@dataclass(frozen=True)
class PauliTerm:
    coefficient: float
    pauli: str

    def to_dict(self) -> dict[str, Any]:
        return {"coefficient": self.coefficient, "pauli": self.pauli}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PauliTerm":
        return cls(coefficient=float(data["coefficient"]), pauli=str(data["pauli"]))


@dataclass(frozen=True)
class PauliHamiltonian:
    n_qubits: int
    terms: tuple[PauliTerm, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        for term in self.terms:
            if len(term.pauli) != self.n_qubits:
                raise ValueError(f"Pauli string {term.pauli!r} length does not match n_qubits={self.n_qubits}")
            invalid = set(term.pauli) - set(_PAULI_MATRICES)
            if invalid:
                raise ValueError(f"Pauli string {term.pauli!r} contains invalid labels: {sorted(invalid)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "terms": [term.to_dict() for term in self.terms],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PauliHamiltonian":
        return cls(
            n_qubits=int(data["n_qubits"]),
            terms=tuple(PauliTerm.from_dict(item) for item in data["terms"]),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dense_matrix(self) -> np.ndarray:
        dimension = 2**self.n_qubits
        matrix = np.zeros((dimension, dimension), dtype=complex)
        for term in self.terms:
            product = np.array([[1]], dtype=complex)
            for label in term.pauli:
                product = np.kron(product, _PAULI_MATRICES[label])
            matrix = matrix + term.coefficient * product
        return matrix

    def ground_energy(self) -> float:
        eigenvalues = np.linalg.eigvalsh(self.to_dense_matrix())
        return float(np.min(eigenvalues).real)

    def expectation(self, state: np.ndarray) -> float:
        vector = np.asarray(state, dtype=complex)
        expected_shape = (2**self.n_qubits,)
        if vector.shape != expected_shape:
            raise ValueError(f"state shape {vector.shape} does not match expected {expected_shape}")
        norm = np.linalg.norm(vector)
        if not np.isclose(norm, 1.0):
            raise ValueError(f"state norm must be 1.0, got {norm}")
        value = np.vdot(vector, self.to_dense_matrix() @ vector)
        return float(value.real)
