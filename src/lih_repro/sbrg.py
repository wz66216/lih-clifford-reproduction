"""SBRG (Spectrum Bifurcation Renormalization Group) baseline adapter.

The SBRG library (github.com/hongyehu/SBRG) is an optional dependency.
When absent, all public functions raise SBRGUnavailable.
When present, this module converts PauliHamiltonian objects into SBRG
Model objects and runs the SBRG flow to produce a baseline energy.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from lih_repro.pauli import PauliHamiltonian


class SBRGUnavailable(RuntimeError):
    """Raised when SBRG is required but the library is not installed."""


def _import_sbrg():
    """Import the SBRG library or raise SBRGUnavailable with a clear message.

    Uses find_spec first for a clean "not installed" path, then attempts
    the actual import so that broken installs are also caught.
    """
    if importlib.util.find_spec("SBRG") is None:
        raise SBRGUnavailable(
            "SBRG library is not installed. "
            "Clone github.com/hongyehu/SBRG and install dependencies via conda."
        )
    import SBRG  # type: ignore[import-not-found]

    return SBRG


def _sbrg_available() -> bool:
    """Return True if the SBRG library can be imported."""
    try:
        _import_sbrg()
        return True
    except SBRGUnavailable:
        return False


_PAULI_TO_SBRG: dict[str, int] = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def pauli_to_sbrg_model(hamiltonian: PauliHamiltonian) -> Any:
    """Convert a PauliHamiltonian to an SBRG Model object."""
    _sbrg = _import_sbrg()

    terms = []
    for term in hamiltonian.terms:
        mu = [_PAULI_TO_SBRG[label] for label in term.pauli]
        mat = _sbrg.mkMat(mu)
        terms.append(_sbrg.Term(mat, val=float(term.coefficient)))

    return _sbrg.Model(size=hamiltonian.n_qubits, terms=terms)


def compute_sbrg_baseline(hamiltonian: PauliHamiltonian) -> dict[str, Any]:
    """Run SBRG on a PauliHamiltonian and return a baseline energy dict."""
    _sbrg = None
    try:
        _sbrg = _import_sbrg()
        model = pauli_to_sbrg_model(hamiltonian)
        sbrg_instance = _sbrg.SBRG(model)
        sbrg_instance.run()
    except SBRGUnavailable:
        raise
    except Exception as exc:
        return {
            "energy": None,
            "status": "failed",
            "n_terms_in": len(hamiltonian.terms),
            "n_terms_out": None,
            "sbrg_version": getattr(_sbrg, "__version__", None),
            "error": str(exc),
        }

    ground_energy: float | None = None
    n_terms_out = 0
    if hasattr(sbrg_instance, "Heff") and sbrg_instance.Heff is not None:
        heff_terms = sbrg_instance.Heff.terms
        n_terms_out = len(heff_terms) if heff_terms else 0
        if n_terms_out > 0:
            ground_energy = min(float(t.val) for t in heff_terms)

    return {
        "energy": ground_energy,
        "status": "ok",
        "n_terms_in": len(hamiltonian.terms),
        "n_terms_out": n_terms_out,
        "sbrg_version": getattr(_sbrg, "__version__", None),
    }
