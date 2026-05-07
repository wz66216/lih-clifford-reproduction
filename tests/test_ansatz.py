import numpy as np
import pytest

from lih_repro.ansatz import (
    CircuitSpec,
    _two_qubit_gate,
    brickwork_pairs,
    gates_per_layer,
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
    spec = CircuitSpec(n_qubits=1, layers=0, clifford_choices=(), rz_sites=(0,))
    theta = np.array([0.321], dtype=float)

    state = run_ansatz_state(spec, theta)

    assert np.isclose(np.linalg.norm(state), 1.0)


def test_brickwork_pairs_n8():
    pairs = brickwork_pairs(8)
    assert len(pairs) == 8
    assert pairs[:4] == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert pairs[4:] == ((1, 2), (3, 4), (5, 6), (7, 0))


def test_gates_per_layer():
    assert gates_per_layer(1) == 0
    assert gates_per_layer(2) == 2
    assert gates_per_layer(8) == 8


def test_two_qubit_gate_is_unitary():
    I4 = np.eye(4, dtype=complex)
    for c in range(16):
        g = _two_qubit_gate(c)
        assert np.allclose(g @ g.conj().T, I4)


def test_two_qubit_gate_exp_pi_zz_over_4():
    """choice 15 = ZZ. exp(i*pi*ZZ/4) = diag(e^{i*pi/4}, e^{-i*pi/4}, e^{-i*pi/4}, e^{i*pi/4})."""
    g = _two_qubit_gate(15)
    expected = np.diag([
        np.exp(1j * np.pi / 4),
        np.exp(-1j * np.pi / 4),
        np.exp(-1j * np.pi / 4),
        np.exp(1j * np.pi / 4),
    ])
    assert np.allclose(g, expected)


def test_choice_zero_is_identity_up_to_phase():
    """choice 0 = II. exp(i*pi*II/4) = e^{i*pi/4} * I."""
    g = _two_qubit_gate(0)
    phase = np.exp(1j * np.pi / 4)
    assert np.allclose(g, phase * np.eye(4))


def test_brickwork_layer_preserves_norm():
    spec = CircuitSpec(n_qubits=2, layers=1, clifford_choices=(5, 3), rz_sites=())
    state = run_ansatz_state(spec, np.array([], dtype=float))
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_two_layers_preserve_norm():
    spec = CircuitSpec(n_qubits=4, layers=2, clifford_choices=tuple(range(8)), rz_sites=())
    state = run_ansatz_state(spec, np.array([], dtype=float))
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_rz_between_layers():
    spec = CircuitSpec(
        n_qubits=2, layers=2,
        clifford_choices=(0, 0, 0, 0),
        rz_sites=(0,), rz_layer=1,
    )
    state = run_ansatz_state(spec, np.array([np.pi / 2], dtype=float))
    assert np.isclose(np.linalg.norm(state), 1.0)
    assert state.shape == (4,)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_qubits": 0, "layers": 0, "clifford_choices": (), "rz_sites": ()}, "n_qubits"),
        ({"n_qubits": 2.0, "layers": 0, "clifford_choices": (), "rz_sites": ()}, "n_qubits"),
        ({"n_qubits": True, "layers": 0, "clifford_choices": (), "rz_sites": ()}, "n_qubits"),
        ({"n_qubits": 2, "layers": -1, "clifford_choices": (), "rz_sites": ()}, "layers"),
        ({"n_qubits": 2, "layers": 0.0, "clifford_choices": (), "rz_sites": ()}, "layers"),
        ({"n_qubits": 2, "layers": True, "clifford_choices": (), "rz_sites": ()}, "layers"),
        ({"n_qubits": 2, "layers": 1, "clifford_choices": (0, 16), "rz_sites": ()}, "clifford"),
        ({"n_qubits": 2, "layers": 1, "clifford_choices": (0, 1.0), "rz_sites": ()}, "clifford"),
        ({"n_qubits": 2, "layers": 1, "clifford_choices": (0, True), "rz_sites": ()}, "clifford"),
        ({"n_qubits": 2, "layers": 1, "clifford_choices": (0, 0), "rz_sites": (2,)}, "rz site"),
        ({"n_qubits": 2, "layers": 1, "clifford_choices": (0, 0), "rz_sites": (1.0,)}, "rz site"),
        ({"n_qubits": 2, "layers": 1, "clifford_choices": (0, 0), "rz_sites": (True,)}, "rz site"),
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
    spec = CircuitSpec(n_qubits=2, layers=1, clifford_choices=(0, 0), rz_sites=(0, 0))

    with pytest.raises(ValueError, match=match):
        run_ansatz_state(spec, theta)


def test_rz_on_first_qubit_uses_correct_phase():
    spec = CircuitSpec(n_qubits=2, layers=0, clifford_choices=(), rz_sites=(0,))

    state = run_ansatz_state(spec, np.array([np.pi], dtype=float))

    expected = np.array([np.exp(-0.5j * np.pi), 0.0, 0.0, 0.0], dtype=complex)
    assert np.allclose(state, expected)
