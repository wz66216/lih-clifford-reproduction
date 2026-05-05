import numpy as np
import pytest

from lih_repro.ansatz import (
    CircuitSpec,
    _apply_cz,
    _apply_single_qubit_gate,
    _apply_two_qubit_clifford,
    _single_gate_for_choice,
    initial_zero_state,
    run_ansatz_state,
)


def test_initial_zero_state_has_unit_norm_and_shape():
    state = initial_zero_state(3)

    assert state.shape == (8,)
    assert np.isclose(np.linalg.norm(state), 1.0)
    assert state.dtype == complex


def test_ansatz_without_gates_returns_zero_state():
    spec = CircuitSpec(n_qubits=2, layers=0, clifford_choices=(), rz_sites=())
    theta = np.array([], dtype=float)

    state = run_ansatz_state(spec, theta)

    assert np.allclose(state, initial_zero_state(2))


def test_single_rz_preserves_norm():
    spec = CircuitSpec(n_qubits=1, layers=1, clifford_choices=(), rz_sites=(0,))
    theta = np.array([0.321], dtype=float)

    state = run_ansatz_state(spec, theta)

    assert np.isclose(np.linalg.norm(state), 1.0)


def test_single_gate_for_choice_returns_plan_gates():
    i = np.array([[1, 0], [0, 1]], dtype=complex)
    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    s = np.array([[1, 0], [0, 1j]], dtype=complex)
    gates = (i, h, s, h @ s)

    for choice in range(16):
        first, second = _single_gate_for_choice(choice)
        assert np.allclose(first, gates[choice % 4])
        assert np.allclose(second, gates[(choice // 4) % 4])


def test_ansatz_applies_rz_after_all_clifford_layers():
    spec = CircuitSpec(n_qubits=2, layers=2, clifford_choices=(0, 0), rz_sites=(0,))
    theta = np.array([np.pi / 3], dtype=float)

    state = run_ansatz_state(spec, theta)

    expected = initial_zero_state(2)
    expected = _apply_two_qubit_clifford(expected, 2, 0, 1, 0)
    expected = _apply_two_qubit_clifford(expected, 2, 0, 1, 0)
    expected = _apply_single_qubit_gate(expected, 2, 0, np.array([[np.exp(-0.5j * theta[0]), 0.0], [0.0, np.exp(0.5j * theta[0])]], dtype=complex))

    assert np.allclose(state, expected)


def test_apply_cz_uses_plan_bit_order():
    state = np.zeros(4, dtype=complex)
    state[3] = 1.0

    updated = _apply_cz(state, 2, 0, 1)

    assert np.allclose(updated, np.array([0.0, 0.0, 0.0, -1.0], dtype=complex))
