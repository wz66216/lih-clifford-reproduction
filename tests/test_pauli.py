import numpy as np

from lih_repro.pauli import PauliHamiltonian, PauliTerm


def test_pauli_hamiltonian_serializes_round_trip():
    ham = PauliHamiltonian(
        n_qubits=2,
        terms=(
            PauliTerm(0.5, "ZI"),
            PauliTerm(-1.25, "XX"),
        ),
        metadata={"distance_angstrom": 1.6, "source": "unit-test"},
    )

    restored = PauliHamiltonian.from_dict(ham.to_dict())

    assert restored.n_qubits == 2
    assert restored.terms == ham.terms
    assert restored.metadata == ham.metadata


def test_dense_matrix_for_single_z_has_expected_eigenvalues():
    ham = PauliHamiltonian(n_qubits=1, terms=(PauliTerm(2.0, "Z"),), metadata={})

    matrix = ham.to_dense_matrix()

    assert np.allclose(matrix, np.array([[2.0, 0.0], [0.0, -2.0]], dtype=complex))
    assert np.isclose(ham.ground_energy(), -2.0)


def test_expectation_value_for_basis_state():
    ham = PauliHamiltonian(n_qubits=1, terms=(PauliTerm(1.5, "Z"),), metadata={})
    state = np.array([0.0, 1.0], dtype=complex)


    assert np.isclose(ham.expectation(state), -1.5)


def test_expectation_matches_dense_matrix_for_mixed_pauli_terms():
    ham = PauliHamiltonian(
        n_qubits=3,
        terms=(
            PauliTerm(0.7, "XIZ"),
            PauliTerm(-0.2, "IYY"),
            PauliTerm(1.1, "ZZI"),
            PauliTerm(-0.4, "III"),
        ),
        metadata={},
    )
    raw_state = np.array([1, 1j, -0.5, 0.25j, 0.75, -1j, 0.5j, -0.25], dtype=complex)
    state = raw_state / np.linalg.norm(raw_state)

    dense_value = np.vdot(state, ham.to_dense_matrix() @ state).real

    assert np.isclose(ham.expectation(state), dense_value)


def test_expectation_does_not_construct_dense_matrix(monkeypatch):
    ham = PauliHamiltonian(n_qubits=2, terms=(PauliTerm(1.0, "XX"),), metadata={})
    state = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)

    def fail_if_called(self):
        raise AssertionError("expectation should evaluate Pauli terms directly")

    monkeypatch.setattr(PauliHamiltonian, "to_dense_matrix", fail_if_called)

    assert np.isclose(ham.expectation(state), 1.0)
