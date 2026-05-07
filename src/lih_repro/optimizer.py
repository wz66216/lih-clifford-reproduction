"""Discrete + continuous optimizer for the Clifford+kRz ansatz.

Implements arXiv:2308.11616 Algorithm 1: at each greedy iteration, draw a
uniform random index over the joint slot space ``Z_{16}^{n*L} x Z_n^k``,
exhaustively scan all candidate values for that slot, and replace it with
the argmin. The continuous angles are re-optimized for every candidate via
multi-start L-BFGS-B (the marginalized cost ``f(X) = min_theta f(X,theta)``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from scipy.optimize import minimize

from lih_repro.ansatz import CircuitSpec, gates_per_layer, run_ansatz_state
from lih_repro.pauli import PauliHamiltonian


@dataclass(frozen=True)
class OptimizerConfig:
    seed: int
    continuous_starts: int
    greedy_iterations: int
    n_init: int = 1
    rz_layer: int = -1  # -1 -> auto: min(1, layers), matching paper Fig.ansatz for L>=2

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
        if not isinstance(self.n_init, int) or isinstance(self.n_init, bool):
            raise TypeError("n_init must be an integer")
        if self.n_init < 1:
            raise ValueError("n_init must be >= 1")
        if not isinstance(self.rz_layer, int) or isinstance(self.rz_layer, bool):
            raise TypeError("rz_layer must be an integer")


@dataclass(frozen=True)
class OptimizationResult:
    energy: float
    circuit: dict[str, Any]
    theta: Tuple[float, ...]
    trace: Tuple[dict[str, Any], ...]


def variational_energy(hamiltonian, n_qubits, layers, clifford_choices, rz_sites, theta, rz_layer: int = -1) -> float:
    spec = CircuitSpec(
        n_qubits=n_qubits,
        layers=layers,
        clifford_choices=tuple(int(x) for x in clifford_choices),
        rz_sites=tuple(int(x) for x in rz_sites),
        rz_layer=rz_layer,
    )
    state = run_ansatz_state(spec, theta=tuple(float(x) for x in theta))
    return hamiltonian.expectation(state)


def _resolve_rz_layer(config_rz_layer: int, layers: int) -> int:
    return config_rz_layer if config_rz_layer >= 0 else min(1, layers)


def _run_single_restart(args: tuple) -> OptimizationResult:
    """Worker for a single n_init restart (picklable top-level function for multiprocessing)."""
    hamiltonian, k, layers, config, seed = args
    rng = np.random.default_rng(seed)
    return _greedy_optimize(hamiltonian, k, layers, config, rng)


def optimize_for_k(hamiltonian: PauliHamiltonian, k: int, layers: int, config: OptimizerConfig) -> OptimizationResult:
    """Single-process version for testing; cli.py handles parallel dispatch in production."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if k < 0:
        raise ValueError("k must be >= 0")
    if not isinstance(layers, int) or isinstance(layers, bool):
        raise TypeError("layers must be an integer")
    if layers < 0:
        raise ValueError("layers must be >= 0")

    best: OptimizationResult | None = None
    for init_idx in range(config.n_init):
        seed = config.seed + 1009 * k + 9173 * hamiltonian.n_qubits + 7919 * init_idx
        rng = np.random.default_rng(seed)
        result = _greedy_optimize(hamiltonian, k, layers, config, rng)
        if best is None or result.energy < best.energy:
            best = result
    assert best is not None
    return best


def _greedy_optimize(
    hamiltonian: PauliHamiltonian,
    k: int,
    layers: int,
    config: OptimizerConfig,
    rng: np.random.Generator,
) -> OptimizationResult:
    n = hamiltonian.n_qubits
    per_layer = gates_per_layer(n)
    n_cliffords = layers * per_layer
    rz_layer = _resolve_rz_layer(config.rz_layer, layers)

    cliffords = tuple(int(x) for x in rng.integers(0, 16, size=n_cliffords))
    rz_sites = tuple(int(x) for x in rng.integers(0, n, size=k)) if k > 0 else ()

    theta, energy = _optimize_theta(hamiltonian, layers, cliffords, rz_sites, rz_layer, config, rng)
    trace: list[dict[str, Any]] = [{"iteration": 0, "kind": "initial", "energy": energy}]

    total_slots = n_cliffords + k
    for iteration in range(1, config.greedy_iterations + 1):
        if total_slots == 0:
            break
        j = int(rng.integers(0, total_slots))
        previous_energy = energy

        if j < n_cliffords:
            best_choice = cliffords[j]
            best_theta = theta
            best_energy = energy
            for cand in range(16):
                trial = list(cliffords)
                trial[j] = cand
                cand_theta, cand_energy = _optimize_theta(
                    hamiltonian, layers, tuple(trial), rz_sites, rz_layer, config, rng
                )
                if cand_energy < best_energy:
                    best_choice = cand
                    best_theta = cand_theta
                    best_energy = cand_energy
            cliffords = tuple(best_choice if i == j else c for i, c in enumerate(cliffords))
            theta = best_theta
            energy = best_energy
            trace.append({
                "iteration": iteration,
                "kind": "clifford",
                "index": j,
                "choice": best_choice,
                "previous_energy": previous_energy,
                "energy": energy,
                "accepted": energy < previous_energy - 1e-12,
            })
        else:
            rz_idx = j - n_cliffords
            best_site = rz_sites[rz_idx]
            best_theta = theta
            best_energy = energy
            for site in range(n):
                trial = list(rz_sites)
                trial[rz_idx] = site
                cand_theta, cand_energy = _optimize_theta(
                    hamiltonian, layers, cliffords, tuple(trial), rz_layer, config, rng
                )
                if cand_energy < best_energy:
                    best_site = site
                    best_theta = cand_theta
                    best_energy = cand_energy
            rz_sites = tuple(best_site if i == rz_idx else s for i, s in enumerate(rz_sites))
            theta = best_theta
            energy = best_energy
            trace.append({
                "iteration": iteration,
                "kind": "rz_site",
                "index": rz_idx,
                "site": best_site,
                "previous_energy": previous_energy,
                "energy": energy,
                "accepted": energy < previous_energy - 1e-12,
            })

    return OptimizationResult(
        energy=float(energy),
        circuit={
            "n_qubits": n,
            "layers": layers,
            "rz_layer": rz_layer,
            "clifford_choices": cliffords,
            "rz_sites": rz_sites,
        },
        theta=tuple(float(x) for x in theta),
        trace=tuple(trace),
    )


def _optimize_theta(
    hamiltonian: PauliHamiltonian,
    layers: int,
    cliffords: Tuple[int, ...],
    rz_sites: Tuple[int, ...],
    rz_layer: int,
    config: OptimizerConfig,
    rng: np.random.Generator,
) -> Tuple[Tuple[float, ...], float]:
    if len(rz_sites) == 0:
        energy = variational_energy(
            hamiltonian, hamiltonian.n_qubits, layers, cliffords, rz_sites, (), rz_layer=rz_layer
        )
        return (), float(energy)

    best_theta: Tuple[float, ...] = tuple(0.0 for _ in rz_sites)
    best_energy = float("inf")
    bounds = [(0.0, 2.0 * np.pi) for _ in rz_sites]

    for _start in range(config.continuous_starts):
        initial = rng.uniform(0.0, 2.0 * np.pi, size=len(rz_sites))

        def objective(values: np.ndarray) -> float:
            return variational_energy(
                hamiltonian, hamiltonian.n_qubits, layers, cliffords, rz_sites, values, rz_layer=rz_layer
            )

        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
        if not np.isfinite(result.fun):
            continue
        if not np.all(np.isfinite(result.x)):
            continue
        energy = float(result.fun)
        if energy < best_energy:
            best_energy = energy
            best_theta = tuple(float(x) for x in result.x)

    if not np.isfinite(best_energy):
        # fall back to evaluating at the last successful initial point
        initial = rng.uniform(0.0, 2.0 * np.pi, size=len(rz_sites))
        energy = variational_energy(
            hamiltonian, hamiltonian.n_qubits, layers, cliffords, rz_sites, initial, rz_layer=rz_layer
        )
        best_energy = float(energy)
        best_theta = tuple(float(x) for x in initial)

    return best_theta, best_energy
