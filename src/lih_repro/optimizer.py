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

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if not isinstance(self.continuous_starts, int) or isinstance(self.continuous_starts, bool):
            raise TypeError("continuous_starts must be an integer")
        if self.continuous_starts < 1:
            raise ValueError("continuous_starts must be >= 1")
        if not isinstance(self.greedy_iterations, int) or isinstance(self.greedy_iterations, bool):
            raise TypeError("greedy_iterations must be an integer")
        if self.greedy_iterations < 0:
            raise ValueError("greedy_iterations must be >= 0")


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
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if k < 0:
        raise ValueError("k must be >= 0")
    if not isinstance(layers, int) or isinstance(layers, bool):
        raise TypeError("layers must be an integer")
    if layers < 0:
        raise ValueError("layers must be >= 0")

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
            previous_energy = energy
            accepted = best_energy < previous_energy
            if accepted:
                theta = best_theta
                energy = best_energy
            trace.append({
                "iteration": iteration,
                "kind": "clifford",
                "index": index,
                "choice": cliffords[index],
                "previous_energy": previous_energy,
                "energy": energy,
                "accepted": accepted,
            })
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
            previous_energy = energy
            accepted = best_energy < previous_energy
            if accepted:
                theta = best_theta
                energy = best_energy
            trace.append({
                "iteration": iteration,
                "kind": "rz_site",
                "index": index,
                "site": rz_sites[index],
                "previous_energy": previous_energy,
                "energy": energy,
                "accepted": accepted,
            })

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
        if not result.success:
            continue
        if not np.isfinite(result.fun):
            continue
        if not np.all(np.isfinite(result.x)):
            continue
        energy = float(result.fun)
        if energy < best_energy:
            best_energy = energy
            best_theta = tuple(float(x) for x in result.x)

    if not np.isfinite(best_energy):
        raise RuntimeError("theta optimization failed to find a finite successful candidate")

    return best_theta, best_energy
