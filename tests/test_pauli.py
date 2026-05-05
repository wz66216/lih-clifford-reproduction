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
