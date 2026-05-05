import numpy as np
import pytest

from lih_repro.optimizer import OptimizerConfig, optimize_for_k, variational_energy
from lih_repro.pauli import PauliHamiltonian, PauliTerm


def test_variational_energy_matches_zero_state_for_z_hamiltonian():
    ham = PauliHamiltonian(n_qubits=1, terms=(PauliTerm(1.0, "Z"),), metadata={})

    energy = variational_energy(ham, n_qubits=1, layers=0, clifford_choices=(), rz_sites=(), theta=())

    assert np.isclose(energy, 1.0)


def test_optimize_for_k_is_reproducible_with_seed():
    ham = PauliHamiltonian(
        n_qubits=2,
        terms=(PauliTerm(0.5, "ZI"), PauliTerm(-0.25, "IZ"), PauliTerm(0.1, "ZZ")),
        metadata={},
    )
    config = OptimizerConfig(seed=7, continuous_starts=1, greedy_iterations=1)

    first = optimize_for_k(ham, k=1, layers=1, config=config)
    second = optimize_for_k(ham, k=1, layers=1, config=config)

    assert first.energy == second.energy
    assert first.circuit == second.circuit
    assert first.theta == second.theta


def test_optimizer_config_validates_inputs():
    with pytest.raises(TypeError, match="seed"):
        OptimizerConfig(seed=True, continuous_starts=1, greedy_iterations=0)

    with pytest.raises(ValueError, match="continuous_starts"):
        OptimizerConfig(seed=1, continuous_starts=0, greedy_iterations=0)

    with pytest.raises(ValueError, match="greedy_iterations"):
        OptimizerConfig(seed=1, continuous_starts=1, greedy_iterations=-1)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"k": -1, "layers": 0}, "k"),
        ({"k": True, "layers": 0}, "k"),
        ({"k": 0, "layers": -1}, "layers"),
        ({"k": 0, "layers": True}, "layers"),
    ],
)
def test_optimize_for_k_validates_k_and_layers(kwargs, match):
    ham = PauliHamiltonian(n_qubits=1, terms=(PauliTerm(1.0, "Z"),), metadata={})
    config = OptimizerConfig(seed=7, continuous_starts=1, greedy_iterations=0)

    with pytest.raises((TypeError, ValueError), match=match):
        optimize_for_k(ham, config=config, **kwargs)


def test_optimize_for_k_returns_useful_trace_for_rz_update():
    ham = PauliHamiltonian(
        n_qubits=2,
        terms=(PauliTerm(0.5, "ZI"), PauliTerm(-0.25, "IZ"), PauliTerm(0.1, "ZZ")),
        metadata={},
    )
    config = OptimizerConfig(seed=11, continuous_starts=1, greedy_iterations=1)

    result = optimize_for_k(ham, k=1, layers=1, config=config)

    assert np.isfinite(result.energy)
    assert len(result.theta) == 1
    assert len(result.trace) == 2
    assert result.trace[0]["kind"] == "initial"
    assert result.trace[1]["kind"] in {"rz_site", "clifford"}
    assert "previous_energy" in result.trace[1]
    assert "accepted" in result.trace[1]
    assert ("site" in result.trace[1]) or ("choice" in result.trace[1])
    assert "energy" in result.trace[1]


def test_optimizer_reproducible_for_nontrivial_case():
    ham = PauliHamiltonian(
        n_qubits=2,
        terms=(PauliTerm(0.5, "ZI"), PauliTerm(-0.25, "IZ"), PauliTerm(0.1, "ZZ")),
        metadata={},
    )
    config = OptimizerConfig(seed=7, continuous_starts=1, greedy_iterations=1)

    first = optimize_for_k(ham, k=1, layers=1, config=config)
    second = optimize_for_k(ham, k=1, layers=1, config=config)

    assert first.energy == second.energy
    assert first.circuit == second.circuit
    assert first.theta == second.theta
