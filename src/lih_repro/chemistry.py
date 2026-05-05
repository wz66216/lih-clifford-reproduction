from __future__ import annotations

import json
import math
import importlib
from pathlib import Path

from lih_repro.pauli import PauliHamiltonian, PauliTerm


class DependencyUnavailable(RuntimeError):
    """Raised when optional chemistry dependencies are required but absent."""


def cache_path_for_distance(cache_dir: Path, distance_angstrom: float) -> Path:
    return Path(cache_dir) / f"lih_{distance_angstrom:.6f}.json"


def _openfermion_available() -> bool:
    try:
        importlib.import_module("openfermion")
        importlib.import_module("openfermionpyscf")
        importlib.import_module("pyscf")
    except (ImportError, ModuleNotFoundError):
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
        try:
            ham = generate_with_openfermion(distance_angstrom)
        except DependencyUnavailable:
            if not allow_synthetic_fixture:
                raise
            ham = synthetic_lih_fixture(distance_angstrom)
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
