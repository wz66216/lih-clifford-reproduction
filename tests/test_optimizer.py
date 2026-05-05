import numpy as np

from lih_repro.optimizer import OptimizerConfig, optimize_for_k, variational_energy
from lih_repro.pauli import PauliHamiltonian, PauliTerm


def test_variational_energy_matches_zero_state_for_z_hamiltonian():
    ham = PauliHamiltonian(n_qubits=1, terms=(PauliTerm(1.0, "Z"),), metadata={})

    energy = variational_energy(ham, n_qubits=1, layers=0, clifford_choices=(), rz_sites=(), theta=())

    assert np.isclose(energy, 1.0)


def test_optimize_for_k_is_reproducible_with_seed():
    ham = PauliHamiltonian(n_qubits=1, terms=(PauliTerm(1.0, "Z"),), metadata={})
    config = OptimizerConfig(seed=7, continuous_starts=2, greedy_iterations=2)

    first = optimize_for_k(ham, k=0, layers=0, config=config)
    second = optimize_for_k(ham, k=0, layers=0, config=config)

    assert first.energy == second.energy
    assert first.circuit == second.circuit
    assert first.theta == second.theta
