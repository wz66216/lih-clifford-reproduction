# LiH Zero-Temperature Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Python reproducibility project that implements the paper-described Clifford + `kRz` optimization structure for LiH zero-temperature energy curves, with `ener.pdf` used only as auxiliary trend validation.

**Architecture:** The project is a small Python package under `src/lih_repro`. Core simulation code is separated from chemistry generation, optimization, plotting, and reporting so each part can be tested independently. The chemistry path uses cached Pauli Hamiltonians and optional OpenFermion/PySCF generation; tests and smoke runs use deterministic fixtures so the package remains verifiable when quantum chemistry dependencies are absent.

**Tech Stack:** Python 3.10+, NumPy, SciPy, Matplotlib, Pytest, optional OpenFermion/PySCF for Hamiltonian generation, standard-library JSON/CSV for configuration and cached artifacts.

---

## File Structure

- Create `pyproject.toml`: package metadata, runtime dependencies, dev dependencies, pytest configuration.
- Create `README.md`: local run instructions and the reproduction boundary statement.
- Create `configs/quick_lih.json`: fast default configuration for one smoke experiment.
- Create `src/lih_repro/__init__.py`: package version export.
- Create `src/lih_repro/pauli.py`: Pauli-string Hamiltonian model, JSON serialization, dense matrix conversion, exact diagonalization.
- Create `src/lih_repro/chemistry.py`: cache-first LiH Hamiltonian loading/generation with explicit dependency failure messages.
- Create `src/lih_repro/ansatz.py`: state-vector Clifford + `kRz` ansatz simulation.
- Create `src/lih_repro/optimizer.py`: continuous angle optimization and randomized greedy discrete search.
- Create `src/lih_repro/figure_reference.py`: auxiliary `ener.pdf` path checks and optional CSV reference loader.
- Create `src/lih_repro/plotting.py`: energy-gap and auxiliary-reference plotting.
- Create `src/lih_repro/report.py`: Markdown report generation with assumptions and limitations.
- Create `src/lih_repro/cli.py`: command-line entry point for the end-to-end experiment.
- Create tests under `tests/` for each focused module.

The workspace is currently not a git repository, so each task ends with a verification checkpoint instead of a mandatory commit. If the implementer initializes git before execution, use the commit command shown in each task.

---

### Task 1: Project Scaffold and Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `configs/quick_lih.json`
- Create: `src/lih_repro/__init__.py`
- Create: `tests/test_package.py`

- [ ] **Step 1: Write the package smoke test**

Create `tests/test_package.py`:

```python
from lih_repro import __version__


def test_package_version_is_string():
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Run the smoke test to verify it fails before scaffolding**

Run:

```powershell
python -m pytest tests/test_package.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lih_repro'`.

- [ ] **Step 3: Create the package scaffold**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "lih-zero-temp-repro"
version = "0.1.0"
description = "Local LiH zero-temperature Clifford+kRz reproduction experiment"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "scipy>=1.10",
  "matplotlib>=3.7"
]

[project.optional-dependencies]
chemistry = [
  "openfermion>=1.6",
  "openfermionpyscf>=0.5",
  "pyscf>=2.3"
]
dev = [
  "pytest>=7.4"
]

[project.scripts]
lih-repro = "lih_repro.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Create `src/lih_repro/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `configs/quick_lih.json`:

```json
{
  "distances_angstrom": [1.4, 2.0, 2.6],
  "k_values": [0, 1],
  "n_qubits": 8,
  "layers": 2,
  "seed": 1234,
  "continuous_starts": 3,
  "greedy_iterations": 4,
  "output_dir": "results/quick_lih",
  "hamiltonian_cache_dir": "data/hamiltonians",
  "reference_pdf": "arXiv-2308.11616v2/figs/ener.pdf",
  "reference_csv": "data/reference/ener_digitized.csv",
  "allow_synthetic_fixture": true
}
```

Create `README.md`:

```markdown
# LiH Zero-Temperature Clifford + kRz Reproduction

This project implements the locally documented algorithmic structure from `Zero and Finite Temperature Quantum Simulations Powered by Quantum Magic` for the LiH zero-temperature `E - E0` curve.

The goal is strict implementation of the described Clifford + `kRz` optimization structure and approximate reproduction of the visual trend in `arXiv-2308.11616v2/figs/ener.pdf`. The project does not claim pointwise reproduction of the paper's hidden numerical data because the local paper files do not provide the exact bond grid, basis, active space, tapering details, SBRG initializer, random seeds, or final optimized circuits.

Install for development:

```powershell
python -m pip install -e ".[dev]"
```

Run the quick smoke experiment:

```powershell
lih-repro --config configs/quick_lih.json
```

If OpenFermion/PySCF are unavailable, the quick configuration may use a deterministic synthetic 8-qubit fixture. Reports generated from that mode are labeled as a software smoke run, not a LiH chemistry reproduction.
```

- [ ] **Step 4: Install the package and run the smoke test**

Run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/test_package.py -v
```

Expected: PASS for `test_package_version_is_string`.

- [ ] **Step 5: Checkpoint**

Run:

```powershell
python -m pytest tests/test_package.py -v
```

Expected: 1 passed.

If git is initialized, commit:

```powershell
git add pyproject.toml README.md configs/quick_lih.json src/lih_repro/__init__.py tests/test_package.py
git commit -m "chore: scaffold LiH reproduction package"
```

---

### Task 2: Pauli Hamiltonian Core

**Files:**
- Create: `src/lih_repro/pauli.py`
- Create: `tests/test_pauli.py`

- [ ] **Step 1: Write failing tests for Pauli terms, dense matrices, and ground energy**

Create `tests/test_pauli.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_pauli.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lih_repro.pauli'`.

- [ ] **Step 3: Implement the Pauli Hamiltonian module**

Create `src/lih_repro/pauli.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@dataclass(frozen=True)
class PauliTerm:
    coefficient: float
    pauli: str

    def to_dict(self) -> dict[str, Any]:
        return {"coefficient": self.coefficient, "pauli": self.pauli}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PauliTerm":
        return cls(coefficient=float(data["coefficient"]), pauli=str(data["pauli"]))


@dataclass(frozen=True)
class PauliHamiltonian:
    n_qubits: int
    terms: tuple[PauliTerm, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        for term in self.terms:
            if len(term.pauli) != self.n_qubits:
                raise ValueError(f"Pauli string {term.pauli!r} length does not match n_qubits={self.n_qubits}")
            invalid = set(term.pauli) - set(_PAULI_MATRICES)
            if invalid:
                raise ValueError(f"Pauli string {term.pauli!r} contains invalid labels: {sorted(invalid)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "terms": [term.to_dict() for term in self.terms],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PauliHamiltonian":
        return cls(
            n_qubits=int(data["n_qubits"]),
            terms=tuple(PauliTerm.from_dict(item) for item in data["terms"]),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dense_matrix(self) -> np.ndarray:
        dimension = 2**self.n_qubits
        matrix = np.zeros((dimension, dimension), dtype=complex)
        for term in self.terms:
            product = np.array([[1]], dtype=complex)
            for label in term.pauli:
                product = np.kron(product, _PAULI_MATRICES[label])
            matrix = matrix + term.coefficient * product
        return matrix

    def ground_energy(self) -> float:
        eigenvalues = np.linalg.eigvalsh(self.to_dense_matrix())
        return float(np.min(eigenvalues).real)

    def expectation(self, state: np.ndarray) -> float:
        vector = np.asarray(state, dtype=complex)
        expected_shape = (2**self.n_qubits,)
        if vector.shape != expected_shape:
            raise ValueError(f"state shape {vector.shape} does not match expected {expected_shape}")
        norm = np.linalg.norm(vector)
        if not np.isclose(norm, 1.0):
            raise ValueError(f"state norm must be 1.0, got {norm}")
        value = np.vdot(vector, self.to_dense_matrix() @ vector)
        return float(value.real)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
python -m pytest tests/test_pauli.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Checkpoint**

Run:

```powershell
python -m pytest tests/test_package.py tests/test_pauli.py -v
```

Expected: 4 passed.

If git is initialized, commit:

```powershell
git add src/lih_repro/pauli.py tests/test_pauli.py
git commit -m "feat: add Pauli Hamiltonian core"
```

---

### Task 3: Hamiltonian Cache and Dependency-Aware Chemistry Interface

**Files:**
- Create: `src/lih_repro/chemistry.py`
- Create: `tests/test_chemistry.py`

- [ ] **Step 1: Write failing tests for cache behavior and dependency failures**

Create `tests/test_chemistry.py`:

```python
import json

import pytest

from lih_repro.chemistry import DependencyUnavailable, load_or_generate_hamiltonian, synthetic_lih_fixture


def test_load_or_generate_reads_existing_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "lih_1.400000.json"
    cache_file.write_text(
        json.dumps(
            {
                "n_qubits": 1,
                "terms": [{"coefficient": -1.0, "pauli": "Z"}],
                "metadata": {"distance_angstrom": 1.4, "source": "cache-test"},
            }
        ),
        encoding="utf-8",
    )

    ham = load_or_generate_hamiltonian(1.4, cache_dir=cache_dir, allow_synthetic_fixture=False)

    assert ham.n_qubits == 1
    assert ham.metadata["source"] == "cache-test"


def test_synthetic_fixture_is_deterministic_8_qubit_hamiltonian():
    first = synthetic_lih_fixture(2.0)
    second = synthetic_lih_fixture(2.0)

    assert first == second
    assert first.n_qubits == 8
    assert first.metadata["source"] == "synthetic-fixture"
    assert first.metadata["distance_angstrom"] == 2.0


def test_generation_without_dependencies_raises_clear_error(tmp_path, monkeypatch):
    import lih_repro.chemistry as chemistry

    monkeypatch.setattr(chemistry, "_openfermion_available", lambda: False)


    with pytest.raises(DependencyUnavailable, match="OpenFermion/PySCF"):
        load_or_generate_hamiltonian(1.4, cache_dir=tmp_path, allow_synthetic_fixture=False)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_chemistry.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lih_repro.chemistry'`.

- [ ] **Step 3: Implement cache-first chemistry module**

Create `src/lih_repro/chemistry.py`:

```python
from __future__ import annotations

import json
import math
from pathlib import Path

from lih_repro.pauli import PauliHamiltonian, PauliTerm


class DependencyUnavailable(RuntimeError):
    """Raised when optional chemistry dependencies are required but absent."""


def cache_path_for_distance(cache_dir: Path, distance_angstrom: float) -> Path:
    return Path(cache_dir) / f"lih_{distance_angstrom:.6f}.json"


def _openfermion_available() -> bool:
    try:
        import openfermion  # noqa: F401
        import openfermionpyscf  # noqa: F401
        import pyscf  # noqa: F401
    except Exception:
        return False
    return True


def load_or_generate_hamiltonian(
    distance_angstrom: float,
    cache_dir: Path,
    allow_synthetic_fixture: bool,
) -> PauliHamiltonian:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path_for_distance(cache_dir, distance_angstrom)
    if path.exists():
        return PauliHamiltonian.from_dict(json.loads(path.read_text(encoding="utf-8")))

    if _openfermion_available():
        ham = generate_with_openfermion(distance_angstrom)
        path.write_text(json.dumps(ham.to_dict(), indent=2), encoding="utf-8")
        return ham

    if allow_synthetic_fixture:
        ham = synthetic_lih_fixture(distance_angstrom)
        path.write_text(json.dumps(ham.to_dict(), indent=2), encoding="utf-8")
        return ham

    raise DependencyUnavailable(
        "OpenFermion/PySCF are unavailable and no cached LiH Hamiltonian exists. "
        "Install with python -m pip install -e \".[chemistry,dev]\" or provide JSON Hamiltonian cache files."
    )


def generate_with_openfermion(distance_angstrom: float) -> PauliHamiltonian:
    raise DependencyUnavailable(
        "OpenFermion/PySCF integration is intentionally isolated. "
        "Use cached Hamiltonians for paper-reproduction runs until the exact basis, active space, and tapering assumptions are selected."
    )


def synthetic_lih_fixture(distance_angstrom: float) -> PauliHamiltonian:
    r = float(distance_angstrom)
    stretch = r - 1.6
    terms = [
        PauliTerm(-7.85 + 0.18 * stretch * stretch, "IIIIIIII"),
        PauliTerm(0.35 * math.exp(-0.45 * r), "ZIIIIIII"),
        PauliTerm(-0.28 * math.exp(-0.30 * r), "IZIIIIII"),
        PauliTerm(0.22 / (1.0 + r), "IIZIIIII"),
        PauliTerm(-0.18 / (1.0 + 0.5 * r), "IIIZIIII"),
        PauliTerm(0.08 * math.exp(-0.20 * r), "XXXXIIII"),
        PauliTerm(0.05 * math.sin(r), "IIXXYYII"),
        PauliTerm(-0.04 * math.cos(0.5 * r), "IIIIZZII"),
        PauliTerm(0.03 * math.exp(-0.10 * r), "IYIYIIII"),
    ]
    return PauliHamiltonian(
        n_qubits=8,
        terms=tuple(terms),
        metadata={
            "distance_angstrom": r,
            "source": "synthetic-fixture",
            "warning": "This deterministic fixture is for software smoke tests and is not a paper LiH Hamiltonian.",
            "n_qubits": 8,
        },
    )
```

- [ ] **Step 4: Run chemistry tests**

Run:

```powershell
python -m pytest tests/test_chemistry.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Checkpoint**

Run:

```powershell
python -m pytest tests/test_package.py tests/test_pauli.py tests/test_chemistry.py -v
```

Expected: 7 passed.

If git is initialized, commit:

```powershell
git add src/lih_repro/chemistry.py tests/test_chemistry.py
git commit -m "feat: add Hamiltonian cache interface"
```

---

### Task 4: Clifford + kRz Ansatz State-Vector Simulator

**Files:**
- Create: `src/lih_repro/ansatz.py`
- Create: `tests/test_ansatz.py`

- [ ] **Step 1: Write failing tests for ansatz state preparation**

Create `tests/test_ansatz.py`:

```python
import numpy as np

from lih_repro.ansatz import CircuitSpec, initial_zero_state, run_ansatz_state


def test_initial_zero_state_has_unit_norm():
    state = initial_zero_state(3)

    assert state.shape == (8,)
    assert np.isclose(np.linalg.norm(state), 1.0)
    assert np.isclose(state[0], 1.0)


def test_ansatz_without_gates_returns_zero_state():
    spec = CircuitSpec(n_qubits=2, layers=0, clifford_choices=(), rz_sites=())

    state = run_ansatz_state(spec, theta=())


    assert np.allclose(state, np.array([1.0, 0.0, 0.0, 0.0], dtype=complex))


def test_single_rz_preserves_norm():
    spec = CircuitSpec(n_qubits=1, layers=0, clifford_choices=(), rz_sites=(0,))
    state = run_ansatz_state(spec, theta=(np.pi / 3,))


    assert np.isclose(np.linalg.norm(state), 1.0)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_ansatz.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lih_repro.ansatz'`.

- [ ] **Step 3: Implement the ansatz simulator**

Create `src/lih_repro/ansatz.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CircuitSpec:
    n_qubits: int
    layers: int
    clifford_choices: tuple[int, ...]
    rz_sites: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = self.layers * max(self.n_qubits - 1, 0)
        if len(self.clifford_choices) != expected:
            raise ValueError(f"expected {expected} Clifford choices, got {len(self.clifford_choices)}")
        for choice in self.clifford_choices:
            if choice < 0 or choice > 15:
                raise ValueError(f"Clifford choice must be in [0, 15], got {choice}")
        for site in self.rz_sites:
            if site < 0 or site >= self.n_qubits:
                raise ValueError(f"Rz site {site} outside n_qubits={self.n_qubits}")


def initial_zero_state(n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    return state


def run_ansatz_state(spec: CircuitSpec, theta: Sequence[float]) -> np.ndarray:
    if len(theta) != len(spec.rz_sites):
        raise ValueError(f"theta length {len(theta)} does not match k={len(spec.rz_sites)}")
    state = initial_zero_state(spec.n_qubits)
    offset = 0
    for _layer in range(spec.layers):
        for left in range(spec.n_qubits - 1):
            state = _apply_two_qubit_clifford(state, spec.n_qubits, left, left + 1, spec.clifford_choices[offset])
            offset += 1
    for angle, site in zip(theta, spec.rz_sites):
        state = _apply_single_qubit_gate(state, spec.n_qubits, site, _rz(angle))
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0):
        state = state / norm
    return state


def _rz(angle: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * angle), 0.0], [0.0, np.exp(0.5j * angle)]],
        dtype=complex,
    )


def _single_gate_for_choice(choice: int) -> tuple[np.ndarray, np.ndarray]:
    i = np.array([[1, 0], [0, 1]], dtype=complex)
    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    s = np.array([[1, 0], [0, 1j]], dtype=complex)
    gates = (i, h, s, h @ s)
    return gates[choice % 4], gates[(choice // 4) % 4]


def _apply_two_qubit_clifford(state: np.ndarray, n_qubits: int, q0: int, q1: int, choice: int) -> np.ndarray:
    first, second = _single_gate_for_choice(choice)
    state = _apply_single_qubit_gate(state, n_qubits, q0, first)
    state = _apply_single_qubit_gate(state, n_qubits, q1, second)
    if choice % 2 == 1:
        state = _apply_cz(state, n_qubits, q0, q1)
    return state


def _apply_single_qubit_gate(state: np.ndarray, n_qubits: int, qubit: int, gate: np.ndarray) -> np.ndarray:
    reshaped = state.reshape([2] * n_qubits)
    moved = np.moveaxis(reshaped, qubit, 0).reshape(2, -1)
    updated = gate @ moved
    restored = np.moveaxis(updated.reshape([2] + [2] * (n_qubits - 1)), 0, qubit)
    return restored.reshape(-1)


def _apply_cz(state: np.ndarray, n_qubits: int, q0: int, q1: int) -> np.ndarray:
    updated = state.copy()
    for index in range(updated.size):
        bit0 = (index >> (n_qubits - 1 - q0)) & 1
        bit1 = (index >> (n_qubits - 1 - q1)) & 1
        if bit0 and bit1:
            updated[index] *= -1
    return updated
```

- [ ] **Step 4: Run ansatz tests**

Run:

```powershell
python -m pytest tests/test_ansatz.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Checkpoint**

Run:

```powershell
python -m pytest tests/test_package.py tests/test_pauli.py tests/test_chemistry.py tests/test_ansatz.py -v
```

Expected: 10 passed.

If git is initialized, commit:

```powershell
git add src/lih_repro/ansatz.py tests/test_ansatz.py
git commit -m "feat: add Clifford kRz ansatz simulator"
```

---

### Task 5: Variational Energy and Randomized Greedy Optimizer

**Files:**
- Create: `src/lih_repro/optimizer.py`
- Create: `tests/test_optimizer.py`

- [ ] **Step 1: Write failing tests for deterministic optimization outputs**

Create `tests/test_optimizer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_optimizer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lih_repro.optimizer'`.

- [ ] **Step 3: Implement optimizer module**

Create `src/lih_repro/optimizer.py`:

```python
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


def variational_energy(
    hamiltonian: PauliHamiltonian,
    n_qubits: int,
    layers: int,
    clifford_choices: Sequence[int],
    rz_sites: Sequence[int],
    theta: Sequence[float],
) -> float:
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
```

- [ ] **Step 4: Run optimizer tests**

Run:

```powershell
python -m pytest tests/test_optimizer.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Checkpoint**

Run:

```powershell
python -m pytest tests/test_package.py tests/test_pauli.py tests/test_chemistry.py tests/test_ansatz.py tests/test_optimizer.py -v
```

Expected: 12 passed.

If git is initialized, commit:

```powershell
git add src/lih_repro/optimizer.py tests/test_optimizer.py
git commit -m "feat: add Clifford kRz optimizer"
```

---

### Task 6: Auxiliary Figure Reference Loader

**Files:**
- Create: `src/lih_repro/figure_reference.py`
- Create: `tests/test_figure_reference.py`

- [ ] **Step 1: Write failing tests for reference metadata and CSV loading**

Create `tests/test_figure_reference.py`:

```python
from lih_repro.figure_reference import load_reference_csv, reference_pdf_status


def test_reference_pdf_status_reports_missing_file(tmp_path):
    status = reference_pdf_status(tmp_path / "missing.pdf")

    assert status["exists"] is False
    assert status["path"].endswith("missing.pdf")


def test_load_reference_csv_reads_named_curves(tmp_path):
    csv_path = tmp_path / "ener_digitized.csv"
    csv_path.write_text(
        "curve,bond_length,energy_gap\nHF,1.4,0.5\nk1,1.4,0.2\n",
        encoding="utf-8",
    )

    curves = load_reference_csv(csv_path)

    assert curves["HF"] == [(1.4, 0.5)]
    assert curves["k1"] == [(1.4, 0.2)]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_figure_reference.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lih_repro.figure_reference'`.

- [ ] **Step 3: Implement figure reference helpers**

Create `src/lih_repro/figure_reference.py`:

```python
from __future__ import annotations

import csv
from pathlib import Path


def reference_pdf_status(path: Path) -> dict[str, object]:
    pdf_path = Path(path)
    return {
        "path": str(pdf_path),
        "exists": pdf_path.exists(),
        "size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "role": "auxiliary visual validation only",
    }


def load_reference_csv(path: Path) -> dict[str, list[tuple[float, float]]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    curves: dict[str, list[tuple[float, float]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            curve = row["curve"]
            point = (float(row["bond_length"]), float(row["energy_gap"]))
            curves.setdefault(curve, []).append(point)
    return curves
```

- [ ] **Step 4: Run figure reference tests**

Run:

```powershell
python -m pytest tests/test_figure_reference.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Checkpoint**

Run:

```powershell
python -m pytest tests/test_package.py tests/test_pauli.py tests/test_chemistry.py tests/test_ansatz.py tests/test_optimizer.py tests/test_figure_reference.py -v
```

Expected: 14 passed.

If git is initialized, commit:

```powershell
git add src/lih_repro/figure_reference.py tests/test_figure_reference.py
git commit -m "feat: add auxiliary figure reference loader"
```

---

### Task 7: Plotting and Report Generation

**Files:**
- Create: `src/lih_repro/plotting.py`
- Create: `src/lih_repro/report.py`
- Create: `tests/test_plotting_report.py`

- [ ] **Step 1: Write failing tests for plot and report outputs**

Create `tests/test_plotting_report.py`:

```python
from lih_repro.plotting import plot_energy_gaps
from lih_repro.report import write_report


def test_plot_energy_gaps_creates_png(tmp_path):
    output = tmp_path / "energy_gap.png"
    rows = [
        {"distance_angstrom": 1.4, "k": 0, "energy_gap": 0.3},
        {"distance_angstrom": 1.4, "k": 1, "energy_gap": 0.1},
        {"distance_angstrom": 2.0, "k": 0, "energy_gap": 0.4},
        {"distance_angstrom": 2.0, "k": 1, "energy_gap": 0.2},
    ]

    plot_energy_gaps(rows, reference_curves={}, output_path=output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_write_report_mentions_reproduction_boundary(tmp_path):
    output = tmp_path / "report.md"
    write_report(
        output_path=output,
        config={"seed": 1234},
        results=[{"distance_angstrom": 1.4, "k": 0, "energy_gap": 0.3}],
        reference_status={"exists": True, "path": "ener.pdf", "role": "auxiliary visual validation only"},
        used_synthetic_fixture=True,
    )

    text = output.read_text(encoding="utf-8")
    assert "not a pointwise reproduction" in text
    assert "synthetic fixture" in text
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_plotting_report.py -v
```

Expected: FAIL with import errors for `lih_repro.plotting` and `lih_repro.report`.

- [ ] **Step 3: Implement plotting module**

Create `src/lih_repro/plotting.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_energy_gaps(
    rows: Iterable[dict[str, float]],
    reference_curves: dict[str, list[tuple[float, float]]],
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["k"]), []).append((float(row["distance_angstrom"]), float(row["energy_gap"])))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, points in sorted(grouped.items()):
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=f"local k={k}")
    for name, points in sorted(reference_curves.items()):
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], linestyle="--", alpha=0.6, label=f"digitized {name}")
    ax.set_xlabel("LiH bond length (Å)")
    ax.set_ylabel("E - E0")
    ax.set_title("LiH Clifford + kRz reproduction: algorithmic trend, not pointwise paper data")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
```

- [ ] **Step 4: Implement report module**

Create `src/lih_repro/report.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(
    output_path: Path,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    reference_status: dict[str, Any],
    used_synthetic_fixture: bool,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LiH Zero-Temperature Reproduction Report",
        "",
        "This report summarizes a strict implementation of the locally documented Clifford + kRz algorithm structure.",
        "It is not a pointwise reproduction of the paper's hidden numerical data.",
        "",
        "## Boundary",
        "",
        "The local paper files omit the exact bond grid, basis, active space, tapering details, SBRG initialization, seeds, and final optimized circuits.",
        "The PDF figure is used as auxiliary visual validation only.",
        "",
        "## Configuration",
        "",
        f"- Seed: {config.get('seed')}",
        f"- Distances: {config.get('distances_angstrom')}",
        f"- k values: {config.get('k_values')}",
        f"- Synthetic fixture used: {used_synthetic_fixture}",
        "",
        "## Reference Figure Status",
        "",
        f"- Path: {reference_status.get('path')}",
        f"- Exists: {reference_status.get('exists')}",
        f"- Role: {reference_status.get('role')}",
        "",
        "## Results",
        "",
        "| distance_angstrom | k | E0 | E | E - E0 | source |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['distance_angstrom']} | {row['k']} | {row['ground_energy']:.10f} | "
            f"{row['energy']:.10f} | {row['energy_gap']:.10f} | {row['source']} |"
        )
    if used_synthetic_fixture:
        lines.extend(
            [
                "",
                "## Synthetic Fixture Warning",
                "",
                "At least one Hamiltonian came from the deterministic synthetic fixture. This validates the software pipeline but is not a LiH chemistry reproduction.",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 5: Run plotting/report tests**

Run:

```powershell
python -m pytest tests/test_plotting_report.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Checkpoint**

Run:

```powershell
python -m pytest tests/test_package.py tests/test_pauli.py tests/test_chemistry.py tests/test_ansatz.py tests/test_optimizer.py tests/test_figure_reference.py tests/test_plotting_report.py -v
```

Expected: 16 passed.

If git is initialized, commit:

```powershell
git add src/lih_repro/plotting.py src/lih_repro/report.py tests/test_plotting_report.py
git commit -m "feat: add plotting and reproduction report"
```

---

### Task 8: End-to-End CLI Experiment Runner

**Files:**
- Create: `src/lih_repro/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI test**

Create `tests/test_cli.py`:

```python
import json

from lih_repro.cli import run_from_config


def test_run_from_config_creates_results_plot_and_report(tmp_path):
    config = {
        "distances_angstrom": [1.4],
        "k_values": [0],
        "n_qubits": 8,
        "layers": 1,
        "seed": 123,
        "continuous_starts": 1,
        "greedy_iterations": 1,
        "output_dir": str(tmp_path / "out"),
        "hamiltonian_cache_dir": str(tmp_path / "cache"),
        "reference_pdf": str(tmp_path / "ener.pdf"),
        "reference_csv": str(tmp_path / "reference.csv"),
        "allow_synthetic_fixture": True,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    outputs = run_from_config(config_path)

    assert outputs["results_json"].exists()
    assert outputs["plot_png"].exists()
    assert outputs["report_md"].exists()
```

- [ ] **Step 2: Run CLI test to verify failure**

Run:

```powershell
python -m pytest tests/test_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lih_repro.cli'`.

- [ ] **Step 3: Implement CLI runner**

Create `src/lih_repro/cli.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lih_repro.chemistry import load_or_generate_hamiltonian
from lih_repro.figure_reference import load_reference_csv, reference_pdf_status
from lih_repro.optimizer import OptimizerConfig, optimize_for_k
from lih_repro.plotting import plot_energy_gaps
from lih_repro.report import write_report


def run_from_config(config_path: Path) -> dict[str, Path]:
    config_path = Path(config_path)
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config["hamiltonian_cache_dir"])
    opt_config = OptimizerConfig(
        seed=int(config["seed"]),
        continuous_starts=int(config["continuous_starts"]),
        greedy_iterations=int(config["greedy_iterations"]),
    )
    results: list[dict[str, Any]] = []
    used_synthetic_fixture = False

    for distance in config["distances_angstrom"]:
        hamiltonian = load_or_generate_hamiltonian(
            float(distance),
            cache_dir=cache_dir,
            allow_synthetic_fixture=bool(config["allow_synthetic_fixture"]),
        )
        source = str(hamiltonian.metadata.get("source", "unknown"))
        if source == "synthetic-fixture":
            used_synthetic_fixture = True
        ground_energy = hamiltonian.ground_energy()
        for k in config["k_values"]:
            result = optimize_for_k(
                hamiltonian,
                k=int(k),
                layers=int(config["layers"]),
                config=opt_config,
            )
            rows = {
                "distance_angstrom": float(distance),
                "k": int(k),
                "ground_energy": ground_energy,
                "energy": result.energy,
                "energy_gap": result.energy - ground_energy,
                "source": source,
                "theta": list(result.theta),
                "circuit": result.circuit,
                "trace": list(result.trace),
            }
            results.append(rows)

    results_json = output_dir / "results.json"
    plot_png = output_dir / "energy_gap.png"
    report_md = output_dir / "report.md"
    results_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    reference_curves = load_reference_csv(Path(config["reference_csv"]))
    reference_status = reference_pdf_status(Path(config["reference_pdf"]))
    plot_energy_gaps(results, reference_curves=reference_curves, output_path=plot_png)
    write_report(
        output_path=report_md,
        config=config,
        results=results,
        reference_status=reference_status,
        used_synthetic_fixture=used_synthetic_fixture,
    )
    return {"results_json": results_json, "plot_png": plot_png, "report_md": report_md}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LiH Clifford+kRz reproduction experiment")
    parser.add_argument("--config", required=True, help="Path to JSON configuration")
    args = parser.parse_args(argv)
    outputs = run_from_config(Path(args.config))
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI test**

Run:

```powershell
python -m pytest tests/test_cli.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Run full test suite**

Run:

```powershell
python -m pytest -v
```

Expected: 17 passed.

- [ ] **Step 6: Run quick experiment**

Run:

```powershell
lih-repro --config configs/quick_lih.json
```

Expected: command exits with code 0 and prints paths for `results_json`, `plot_png`, and `report_md` under `results/quick_lih`.

If git is initialized, commit:

```powershell
git add src/lih_repro/cli.py tests/test_cli.py
git commit -m "feat: add end-to-end reproduction runner"
```

---

### Task 9: Final Verification Against Spec

**Files:**
- Verify: `docs/superpowers/specs/2026-05-05-lih-zero-temperature-reproduction-design.md`
- Verify: `results/quick_lih/results.json`
- Verify: `results/quick_lih/energy_gap.png`
- Verify: `results/quick_lih/report.md`

- [ ] **Step 1: Run all tests**

Run:

```powershell
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run the quick experiment from the checked-in config**

Run:

```powershell
lih-repro --config configs/quick_lih.json
```

Expected: command exits with code 0 and produces `results/quick_lih/results.json`, `results/quick_lih/energy_gap.png`, and `results/quick_lih/report.md`.

- [ ] **Step 3: Inspect result artifacts for required boundary language**

Run:

```powershell
python - <<'PY'
from pathlib import Path
report = Path('results/quick_lih/report.md').read_text(encoding='utf-8')
assert 'not a pointwise reproduction' in report
assert 'auxiliary visual validation only' in report
print('report boundary language verified')
PY
```

Expected: prints `report boundary language verified`.

- [ ] **Step 4: Inspect saved optimization data**

Run:

```powershell
python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path('results/quick_lih/results.json').read_text(encoding='utf-8'))
assert rows
for row in rows:
    assert 'distance_angstrom' in row
    assert 'k' in row
    assert 'ground_energy' in row
    assert 'energy' in row
    assert 'energy_gap' in row
    assert 'theta' in row
    assert 'circuit' in row
    assert 'trace' in row
print(f'verified {len(rows)} result rows')
PY
```

Expected: prints `verified N result rows` with `N` greater than 0.

- [ ] **Step 5: Final checkpoint**

If git is initialized, commit final verification artifacts that should be versioned and leave large generated files untracked unless the user requests them:

```powershell
git status --short
git add configs/quick_lih.json README.md pyproject.toml src tests docs/superpowers/specs docs/superpowers/plans
git commit -m "feat: implement LiH zero-temperature reproduction pipeline"
```

Expected in non-git workspace: skip commit and report that the workspace is not a git repository.

---

## Self-Review

### Spec coverage

- LiH zero-temperature `E - E0` only: implemented by Tasks 3, 5, 7, and 8.
- Locally documented Clifford + `kRz` algorithm structure: implemented by Tasks 4 and 5.
- Hamiltonian generation/cache with explicit assumptions: implemented by Task 3 and reported by Task 7.
- `ener.pdf` as auxiliary validation: implemented by Task 6 and integrated in Tasks 7 and 8.
- Reproducibility through seed, trace, parameters, and cached Hamiltonians: implemented by Tasks 3, 5, 8, and verified in Task 9.
- Clear boundary language that results are not pointwise paper data: implemented by README in Task 1 and report generation in Task 7.

### Placeholder scan

The plan contains no unresolved placeholders, incomplete sections, or steps that ask for unspecified behavior. Code snippets define the functions and classes referenced by later tasks.

### Type consistency

The plan consistently uses `PauliHamiltonian`, `PauliTerm`, `CircuitSpec`, `OptimizerConfig`, `OptimizationResult`, `load_or_generate_hamiltonian`, `optimize_for_k`, `plot_energy_gaps`, `write_report`, and `run_from_config` with matching signatures across tests and implementation steps.
