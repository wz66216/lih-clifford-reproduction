import numpy as np
import pytest

from lih_repro.ansatz import (
    CircuitSpec,
    _apply_cz,
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


def test_apply_cz_uses_plan_bit_order():
    state = np.zeros(4, dtype=complex)
    state[3] = 1.0

    updated = _apply_cz(state, 2, 0, 1)

    assert np.allclose(updated, np.array([0.0, 0.0, 0.0, -1.0], dtype=complex))


def test_choice_five_observes_cz_on_two_qubits():
    spec = CircuitSpec(n_qubits=2, layers=1, clifford_choices=(5,), rz_sites=())

    state = run_ansatz_state(spec, np.array([], dtype=float))

    expected = np.array([0.5, 0.5, 0.5, -0.5], dtype=complex)
    assert np.allclose(state, expected)


def test_h_then_s_adds_relative_phase_on_first_qubit():
    spec = CircuitSpec(n_qubits=2, layers=2, clifford_choices=(1, 2), rz_sites=())

    state = run_ansatz_state(spec, np.array([], dtype=float))

    expected = np.array([1 / np.sqrt(2), 0.0, 1j / np.sqrt(2), 0.0], dtype=complex)
    assert np.allclose(state, expected)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_qubits": 0, "layers": 0, "clifford_choices": (), "rz_sites": ()}, "n_qubits"),
        ({"n_qubits": 2.0, "layers": 0, "clifford_choices": (), "rz_sites": ()}, "n_qubits"),
        ({"n_qubits": True, "layers": 0, "clifford_choices": (), "rz_sites": ()}, "n_qubits"),
        ({"n_qubits": 2, "layers": -1, "clifford_choices": (), "rz_sites": ()}, "layers"),
        ({"n_qubits": 2, "layers": 0.0, "clifford_choices": (), "rz_sites": ()}, "layers"),
        ({"n_qubits": 2, "layers": True, "clifford_choices": (), "rz_sites": ()}, "layers"),
        ({"n_qubits": 2, "layers": 1, "clifford_choices": (16,), "rz_sites": ()}, "clifford"),
        ({"n_qubits": 2, "layers": 1, "clifford_choices": (1.0,), "rz_sites": ()}, "clifford"),
        ({"n_qubits": 2, "layers": 1, "clifford_choices": (True,), "rz_sites": ()}, "clifford"),
        ({"n_qubits": 2, "layers": 1, "clifford_choices": (0,), "rz_sites": (2,)}, "rz site"),
        ({"n_qubits": 2, "layers": 1, "clifford_choices": (0,), "rz_sites": (1.0,)}, "rz site"),
        ({"n_qubits": 2, "layers": 1, "clifford_choices": (0,), "rz_sites": (True,)}, "rz site"),
    ],
)
def test_circuit_spec_rejects_invalid_inputs(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        CircuitSpec(**kwargs)


@pytest.mark.parametrize(
    "theta, match",
    [
        (0.1, "1-D"),
        (np.array([[0.1]]), "1-D"),
        (np.array([0.1, np.nan]), "finite"),
        (np.array([0.1, np.inf]), "finite"),
        (np.array([0.1]), "length"),
    ],
)
def test_run_ansatz_state_rejects_invalid_theta(theta, match):
    spec = CircuitSpec(n_qubits=1, layers=1, clifford_choices=(), rz_sites=(0, 0))

    with pytest.raises(ValueError, match=match):
        run_ansatz_state(spec, theta)


def test_choice_one_applies_h_on_first_qubit_and_cz():
    spec = CircuitSpec(n_qubits=2, layers=1, clifford_choices=(1,), rz_sites=())

    state = run_ansatz_state(spec, np.array([], dtype=float))

    expected = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
    assert np.allclose(state, expected)


def test_rz_on_first_qubit_uses_plan_phase():
    spec = CircuitSpec(n_qubits=1, layers=0, clifford_choices=(), rz_sites=(0,))

    state = run_ansatz_state(spec, np.array([np.pi], dtype=float))

    expected = np.array([np.exp(-0.5j * np.pi), 0.0], dtype=complex)
    assert np.allclose(state, expected)
