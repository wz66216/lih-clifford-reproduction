import numpy as np
import pytest

from lih_repro.ansatz import CircuitSpec, initial_zero_state, run_ansatz_state


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
