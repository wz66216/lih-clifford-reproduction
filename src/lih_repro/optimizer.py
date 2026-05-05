from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from lih_repro.ansatz import CircuitSpec, run_ansatz_state
from lih_repro.pauli import PauliHamiltonian


@dataclass(frozen=True)
class OptimizerConfig:
    seed: int
    continuous_starts: int
    greedy_iterations: int


@dataclass(frozen=True)
class OptimizationResult:
    energy: float
    circuit: dict[str, Any]
    theta: tuple[float, ...]
    trace: tuple[dict[str, Any], ...]


def variational_energy(hamiltonian, n_qubits, layers, clifford_choices, rz_sites, theta) -> float:
    spec = CircuitSpec(
        n_qubits=n_qubits,
        layers=layers,
        clifford_choices=tuple(int(x) for x in clifford_choices),
        rz_sites=tuple(int(x) for x in rz_sites),
    )
    state = run_ansatz_state(spec, theta=tuple(float(x) for x in theta))
    return hamiltonian.expectation(state)


def optimize_for_k(hamiltonian: PauliHamiltonian, k: int, layers: int, config: OptimizerConfig) -> OptimizationResult:
    rng = np.random.default_rng(config.seed + 1009 * k + 9173 * hamiltonian.n_qubits)
    n_cliffords = layers * max(hamiltonian.n_qubits - 1, 0)
    cliffords = tuple(int(x) for x in rng.integers(0, 16, size=n_cliffords))
    rz_sites = tuple(int(x) for x in rng.integers(0, hamiltonian.n_qubits, size=k))
    theta, energy = _optimize_theta(hamiltonian, layers, cliffords, rz_sites, config, rng)
    trace: list[dict[str, Any]] = [{"iteration": 0, "energy": energy, "kind": "initial"}]

    for iteration in range(1, config.greedy_iterations + 1):
        if n_cliffords > 0 and (k == 0 or rng.random() < 0.7):
            index = int(rng.integers(0, n_cliffords))
            best_choice = cliffords[index]
            best_theta = theta
            best_energy = energy
            for candidate in range(16):
                candidate_cliffords = list(cliffords)
                candidate_cliffords[index] = candidate
                candidate_theta, candidate_energy = _optimize_theta(
                    hamiltonian, layers, tuple(candidate_cliffords), rz_sites, config, rng
                )
                if candidate_energy < best_energy:
                    best_choice = candidate
                    best_theta = candidate_theta
                    best_energy = candidate_energy
            cliffords = tuple(best_choice if i == index else value for i, value in enumerate(cliffords))
            theta = best_theta
            energy = best_energy
            trace.append({"iteration": iteration, "energy": energy, "kind": "clifford", "index": index})
        elif k > 0:
            index = int(rng.integers(0, k))
            best_site = rz_sites[index]
            best_theta = theta
            best_energy = energy
            for site in range(hamiltonian.n_qubits):
                candidate_sites = list(rz_sites)
                candidate_sites[index] = site
                candidate_theta, candidate_energy = _optimize_theta(
                    hamiltonian, layers, cliffords, tuple(candidate_sites), config, rng
                )
                if candidate_energy < best_energy:
                    best_site = site
                    best_theta = candidate_theta
                    best_energy = candidate_energy
            rz_sites = tuple(best_site if i == index else value for i, value in enumerate(rz_sites))
            theta = best_theta
            energy = best_energy
            trace.append({"iteration": iteration, "energy": energy, "kind": "rz_site", "index": index})

    return OptimizationResult(
        energy=float(energy),
        circuit={"n_qubits": hamiltonian.n_qubits, "layers": layers, "clifford_choices": cliffords, "rz_sites": rz_sites},
        theta=tuple(float(x) for x in theta),
        trace=tuple(trace),
    )


def _optimize_theta(
    hamiltonian: PauliHamiltonian,
    layers: int,
    cliffords: tuple[int, ...],
    rz_sites: tuple[int, ...],
    config: OptimizerConfig,
    rng: np.random.Generator,
) -> tuple[tuple[float, ...], float]:
    if len(rz_sites) == 0:
        energy = variational_energy(hamiltonian, hamiltonian.n_qubits, layers, cliffords, rz_sites, ())
        return (), float(energy)

    best_theta: tuple[float, ...] = tuple(0.0 for _ in rz_sites)
    best_energy = float("inf")
    bounds = [(0.0, 2.0 * np.pi) for _ in rz_sites]

    for _start in range(config.continuous_starts):
        initial = rng.uniform(0.0, 2.0 * np.pi, size=len(rz_sites))

        def objective(values: np.ndarray) -> float:
            return variational_energy(hamiltonian, hamiltonian.n_qubits, layers, cliffords, rz_sites, values)

        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds, options={"maxiter": 50})
        energy = float(result.fun)
        if energy < best_energy:
            best_energy = energy
            best_theta = tuple(float(x) for x in result.x)

    return best_theta, best_energy
