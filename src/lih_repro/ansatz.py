from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircuitSpec:
    n_qubits: int
    layers: int
    clifford_choices: tuple[int, ...]
    rz_sites: tuple[int, ...]

    def __post_init__(self):
        expected = self.layers * max(self.n_qubits - 1, 0)
        if len(self.clifford_choices) != expected:
            raise ValueError("clifford_choices length does not match layers")
        for choice in self.clifford_choices:
            if not 0 <= choice <= 15:
                raise ValueError("clifford choice must be in 0..15")
        for site in self.rz_sites:
            if not 0 <= site < self.n_qubits:
                raise ValueError("rz site out of range")


def initial_zero_state(n_qubits: int):
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0 + 0.0j
    return state


def _rz(theta):
    return np.array([[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=complex)


def _single_gate_for_choice(choice: int) -> tuple[np.ndarray, np.ndarray]:
    i = np.array([[1, 0], [0, 1]], dtype=complex)
    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    s = np.array([[1, 0], [0, 1j]], dtype=complex)
    gates = (i, h, s, h @ s)
    return gates[choice % 4], gates[(choice // 4) % 4]


def _apply_single_qubit_gate(state: np.ndarray, n_qubits: int, qubit: int, gate: np.ndarray) -> np.ndarray:
    reshaped = state.reshape([2] * n_qubits)
    moved = np.moveaxis(reshaped, qubit, 0).reshape(2, -1)
    updated = gate @ moved
    restored = np.moveaxis(updated.reshape([2] + [2] * (n_qubits - 1)), 0, qubit)
    return restored.reshape(-1)


def _apply_cz(state: np.ndarray, n_qubits: int, q0: int, q1: int) -> np.ndarray:
    out = state.copy()
    for idx in range(out.size):
        bit0 = (idx >> (n_qubits - 1 - q0)) & 1
        bit1 = (idx >> (n_qubits - 1 - q1)) & 1
        if bit0 and bit1:
            out[idx] *= -1
    return out


def _apply_two_qubit_clifford(state: np.ndarray, n_qubits: int, q0: int, q1: int, choice: int) -> np.ndarray:
    first, second = _single_gate_for_choice(choice)
    state = _apply_single_qubit_gate(state, n_qubits, q0, first)
    state = _apply_single_qubit_gate(state, n_qubits, q1, second)
    if choice % 2 == 1:
        state = _apply_cz(state, n_qubits, q0, q1)
    return state


def run_ansatz_state(spec: CircuitSpec, theta):
    theta = np.asarray(theta, dtype=float)
    if len(theta) != len(spec.rz_sites):
        raise ValueError("theta length does not match rz_sites")

    state = initial_zero_state(spec.n_qubits)
    choice_idx = 0
    for _layer in range(spec.layers):
        for q in range(spec.n_qubits - 1):
            state = _apply_two_qubit_clifford(state, spec.n_qubits, q, q + 1, spec.clifford_choices[choice_idx])
            choice_idx += 1
    for angle, site in zip(theta, spec.rz_sites):
        state = _apply_single_qubit_gate(state, spec.n_qubits, site, _rz(angle))

    norm = np.linalg.norm(state)
    return state if norm == 0 else state / norm
