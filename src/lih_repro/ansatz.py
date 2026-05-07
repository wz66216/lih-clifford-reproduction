"""Brickwork Clifford + kRz ansatz matching arXiv:2308.11616 Fig. ansatz / app:optimization.

A single layer applies n two-qubit gates from the gate set
``{ exp(i*pi*P/4) : P in {I,X,Y,Z}^{tensor 2} }`` (16 elements) on the n
nearest-neighbor pairs of a periodic chain. The pairs are split into an
even sublayer ``{(0,1),(2,3),...}`` and an odd sublayer
``{(1,2),(3,4),...,(n-1,0)}``. The k Rz gates are inserted at one fixed
inter-layer position (parameterized by ``rz_layer``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Tuple

import numpy as np


_PAULI_1Q: Tuple[np.ndarray, ...] = (
    np.array([[1, 0], [0, 1]], dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


@lru_cache(maxsize=None)
def _two_qubit_gate(choice: int) -> np.ndarray:
    """Return the 4x4 unitary exp(i*pi*P/4) for choice in 0..15.

    Encoding: choice = p0_idx * 4 + p1_idx with p_idx in 0..3 mapping to (I,X,Y,Z).
    Since each Pauli squares to identity, exp(i*pi*P/4) = (I + i*P)/sqrt(2).
    """
    if not 0 <= choice <= 15:
        raise ValueError(f"clifford choice must be in 0..15, got {choice}")
    p0_idx, p1_idx = choice // 4, choice % 4
    pauli = np.kron(_PAULI_1Q[p0_idx], _PAULI_1Q[p1_idx])
    return (np.eye(4, dtype=complex) + 1j * pauli) / np.sqrt(2.0)


def gates_per_layer(n_qubits: int) -> int:
    """Number of two-qubit gates per brickwork layer (even + odd sublayers, periodic)."""
    return n_qubits if n_qubits >= 2 else 0


def brickwork_pairs(n_qubits: int) -> Tuple[Tuple[int, int], ...]:
    """Return the ordered list of (q0, q1) pairs for one brickwork layer.

    Even sublayer first, then odd sublayer with periodic boundary on the last pair.
    Length equals ``gates_per_layer(n_qubits)``.
    """
    if n_qubits < 2:
        return ()
    even = tuple((i, i + 1) for i in range(0, n_qubits - 1, 2))
    odd = tuple((i, i + 1) for i in range(1, n_qubits - 1, 2)) + ((n_qubits - 1, 0),)
    return even + odd


@dataclass(frozen=True)
class CircuitSpec:
    n_qubits: int
    layers: int
    clifford_choices: Tuple[int, ...]
    rz_sites: Tuple[int, ...]
    rz_layer: int = field(default=-1)

    def __post_init__(self):
        if not isinstance(self.n_qubits, int) or isinstance(self.n_qubits, bool):
            raise TypeError("n_qubits must be an integer")
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be > 0")
        if not isinstance(self.layers, int) or isinstance(self.layers, bool):
            raise TypeError("layers must be an integer")
        if self.layers < 0:
            raise ValueError("layers must be >= 0")
        if not isinstance(self.rz_layer, int) or isinstance(self.rz_layer, bool):
            raise TypeError("rz_layer must be an integer")
        expected_cliffords = self.layers * gates_per_layer(self.n_qubits)
        if len(self.clifford_choices) != expected_cliffords:
            raise ValueError(
                f"clifford_choices length {len(self.clifford_choices)} does not match "
                f"layers*gates_per_layer={expected_cliffords}"
            )
        for choice in self.clifford_choices:
            if not isinstance(choice, int) or isinstance(choice, bool):
                raise TypeError("clifford choice must be an integer")
            if not 0 <= choice <= 15:
                raise ValueError("clifford choice must be in 0..15")
        for site in self.rz_sites:
            if not isinstance(site, int) or isinstance(site, bool):
                raise TypeError("rz site must be an integer")
            if not 0 <= site < self.n_qubits:
                raise ValueError("rz site out of range")
        rz_layer = self.rz_layer if self.rz_layer >= 0 else self.layers
        if not 0 <= rz_layer <= self.layers:
            raise ValueError(f"rz_layer must be in 0..layers, got {self.rz_layer}")
        object.__setattr__(self, "rz_layer", rz_layer)


def initial_zero_state(n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0 + 0.0j
    return state


def _apply_two_qubit_gate(state: np.ndarray, n: int, q0: int, q1: int, gate: np.ndarray) -> np.ndarray:
    """Apply a 4x4 unitary to the (q0, q1) sub-system of an n-qubit state vector.

    Convention: qubit 0 is the most significant bit (matches the existing PauliHamiltonian
    dense matrix construction). The reshape maps the q0-axis to the slow-varying index
    of the 4-dim block, matching ``np.kron(P_q0, P_q1)``.
    """
    if q0 == q1:
        raise ValueError("two-qubit gate requires distinct qubits")
    arr = state.reshape([2] * n)
    arr = np.moveaxis(arr, [q0, q1], [0, 1])
    rest_shape = arr.shape[2:]
    flat = arr.reshape(4, -1)
    flat = gate @ flat
    arr = flat.reshape((2, 2) + rest_shape)
    arr = np.moveaxis(arr, [0, 1], [q0, q1])
    return arr.reshape(-1)


def _apply_rz(state: np.ndarray, n: int, qubit: int, theta: float) -> np.ndarray:
    arr = state.reshape([2] * n).copy()
    arr = np.moveaxis(arr, qubit, 0)
    rest_shape = arr.shape[1:]
    flat = arr.reshape(2, -1)
    out = np.empty_like(flat)
    out[0] = np.exp(-0.5j * theta) * flat[0]
    out[1] = np.exp(+0.5j * theta) * flat[1]
    out = out.reshape((2,) + rest_shape)
    out = np.moveaxis(out, 0, qubit)
    return out.reshape(-1)


def _apply_layer(state: np.ndarray, n: int, layer_choices: Tuple[int, ...]) -> np.ndarray:
    pairs = brickwork_pairs(n)
    for (q0, q1), choice in zip(pairs, layer_choices):
        state = _apply_two_qubit_gate(state, n, q0, q1, _two_qubit_gate(int(choice)))
    return state


def _apply_rz_block(state: np.ndarray, n: int, rz_sites: Tuple[int, ...], theta: np.ndarray) -> np.ndarray:
    for angle, site in zip(theta, rz_sites):
        state = _apply_rz(state, n, int(site), float(angle))
    return state


def run_ansatz_state(spec: CircuitSpec, theta) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1:
        raise ValueError("theta must be 1-D")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must contain only finite values")
    if len(theta) != len(spec.rz_sites):
        raise ValueError("theta length does not match rz_sites")

    n = spec.n_qubits
    per_layer = gates_per_layer(n)
    state = initial_zero_state(n)

    for l in range(spec.layers):
        if l == spec.rz_layer:
            state = _apply_rz_block(state, n, spec.rz_sites, theta)
        layer_choices = spec.clifford_choices[l * per_layer : (l + 1) * per_layer]
        state = _apply_layer(state, n, layer_choices)

    if spec.rz_layer == spec.layers:
        state = _apply_rz_block(state, n, spec.rz_sites, theta)

    norm = np.linalg.norm(state)
    return state if norm == 0 else state / norm
