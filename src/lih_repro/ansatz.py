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


def _single_gate_for_choice(choice: int):
    gates = {
        0: np.eye(2, dtype=complex),
        1: np.array([[0, 1], [1, 0]], dtype=complex),
        2: np.array([[0, -1j], [1j, 0]], dtype=complex),
        3: np.array([[1, 0], [0, -1]], dtype=complex),
    }
    return gates.get(choice % 4, np.eye(2, dtype=complex))


def _apply_single_qubit_gate(state, gate, qubit, n_qubits):
    out = state.copy()
    step = 2 ** qubit
    period = step * 2
    for base in range(0, len(state), period):
        for offset in range(step):
            i0 = base + offset
            i1 = i0 + step
            a, b = state[i0], state[i1]
            out[i0] = gate[0, 0] * a + gate[0, 1] * b
            out[i1] = gate[1, 0] * a + gate[1, 1] * b
    return out


def _apply_cz(state, q0, q1, n_qubits):
    out = state.copy()
    for idx in range(len(state)):
        if ((idx >> q0) & 1) and ((idx >> q1) & 1):
            out[idx] *= -1
    return out


def _apply_two_qubit_clifford(state, choice, q0, q1, n_qubits):
    state = _apply_single_qubit_gate(state, _single_gate_for_choice(choice), q0, n_qubits)
    if choice % 2:
        state = _apply_single_qubit_gate(state, _single_gate_for_choice(choice + 1), q1, n_qubits)
    if choice & 4:
        state = _apply_cz(state, q0, q1, n_qubits)
    return state


def run_ansatz_state(spec: CircuitSpec, theta):
    theta = np.asarray(theta, dtype=float)
    if len(theta) != len(spec.rz_sites):
        raise ValueError("theta length does not match rz_sites")

    state = initial_zero_state(spec.n_qubits)
    choice_idx = 0
    for _layer in range(spec.layers):
        for q in range(spec.n_qubits - 1):
            state = _apply_two_qubit_clifford(state, spec.clifford_choices[choice_idx], q, q + 1, spec.n_qubits)
            choice_idx += 1
        for angle, site in zip(theta, spec.rz_sites):
            state = _apply_single_qubit_gate(state, _rz(angle), site, spec.n_qubits)

    norm = np.linalg.norm(state)
    return state if norm == 0 else state / norm
