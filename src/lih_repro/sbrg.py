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

    Finds SBRG first, then attempts the actual import. Both "not installed"
    and "broken install" (missing transitive dependencies) produce
    SBRGUnavailable rather than raw ImportError.
    """
    if importlib.util.find_spec("SBRG") is None:
        raise SBRGUnavailable(
            "SBRG library is not installed. "
            "Clone github.com/hongyehu/SBRG and install dependencies via conda."
        )
    try:
        import SBRG  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SBRGUnavailable(
            f"SBRG library found but failed to import: {exc}. "
            "Check that all conda dependencies (numpy, numba, qutip, scipy) are installed."
        ) from exc

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
        terms.append(_sbrg.Term(mat, float(term.coefficient)))

    model = _sbrg.Model()
    model.size = hamiltonian.n_qubits
    model.terms = terms
    return model


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

    # SBRG.Heff is a Ham object. The RG flow produces a nearly-diagonal
    # effective Hamiltonian; min(t.val) is an approximate ground energy.
    # For rigorous ground energy, the full effective Hamiltonian should be
    # diagonalized or SBRG.grndstate_blk() / SBRG.energy() should be used.
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


# ---------------------------------------------------------------------------
# Pauli string algebra (no SBRG dependency)
# ---------------------------------------------------------------------------


def _pauli_commutes(p1: str, p2: str) -> bool:
    """Check if two Pauli strings commute (True) or anti-commute (False)."""
    anti = 0
    for a, b in zip(p1, p2):
        if a == "I" or b == "I" or a == b:
            continue
        anti += 1
    return anti % 2 == 0


def _multiply_pauli_strings(p1: str, p2: str) -> tuple[str, complex]:
    """Multiply two Pauli strings. Returns (product, phase)."""
    result = []
    phase = 1.0 + 0.0j
    for a, b in zip(p1, p2):
        if a == "I":
            result.append(b)
        elif b == "I":
            result.append(a)
        elif a == b:
            result.append("I")
        elif a == "X" and b == "Y":
            result.append("Z"); phase *= 1j
        elif a == "X" and b == "Z":
            result.append("Y"); phase *= -1j
        elif a == "Y" and b == "X":
            result.append("Z"); phase *= -1j
        elif a == "Y" and b == "Z":
            result.append("X"); phase *= 1j
        elif a == "Z" and b == "X":
            result.append("Y"); phase *= 1j
        elif a == "Z" and b == "Y":
            result.append("X"); phase *= -1j
    return "".join(result), phase


def _sbrg_mat_to_pauli_string(mat, n_qubits: int) -> str:
    """Convert an SBRG Mat object to a Pauli string."""
    chars = []
    for i in range(n_qubits):
        in_x = i in mat.Xs
        in_z = i in mat.Zs
        if in_x and in_z:
            chars.append("Y")
        elif in_x:
            chars.append("X")
        elif in_z:
            chars.append("Z")
        else:
            chars.append("I")
    return "".join(chars)


def _conjugate_pauli_term_by_rotations(
    pauli: str, coeff: float, rcc: list, n_qubits: int
) -> tuple[str, float]:
    q = pauli
    c = complex(coeff, 0.0)
    for r_term in reversed(rcc):
        gen_str = _sbrg_mat_to_pauli_string(r_term.mat, n_qubits)
        if not _pauli_commutes(gen_str, q):
            gen_val = float(r_term.val)
            q, mul_phase = _multiply_pauli_strings(q, gen_str)
            c *= -1j * gen_val
            c *= mul_phase
    return q, c.real


def compute_sbrg_initializer(
    hamiltonian: PauliHamiltonian,
) -> tuple[PauliHamiltonian, dict[str, Any]]:
    """Run SBRG and return a transformed Hamiltonian plus baseline info."""
    from lih_repro.pauli import PauliTerm as PT

    _sbrg = None
    try:
        _sbrg = _import_sbrg()
        model = pauli_to_sbrg_model(hamiltonian)
        sbrg_instance = _sbrg.SBRG(model)
        sbrg_instance.run()
    except SBRGUnavailable:
        raise
    except Exception as exc:
        return hamiltonian, {
            "energy": None,
            "status": "failed",
            "n_terms_in": len(hamiltonian.terms),
            "n_terms_out": None,
            "sbrg_version": getattr(_sbrg, "__version__", None),
            "error": str(exc),
        }

    heff = getattr(sbrg_instance, "Heff", None)
    rcc = getattr(sbrg_instance, "RCC", [])
    nq = hamiltonian.n_qubits

    new_terms = []
    heff_terms = list(getattr(heff, "terms", []) or []) if heff is not None else []
    if heff_terms:
        for t in heff_terms:
            p_str = _sbrg_mat_to_pauli_string(t.mat, nq)
            p_conj, c_conj = _conjugate_pauli_term_by_rotations(p_str, float(t.val), rcc, nq)
            if abs(c_conj) > 1e-14:
                new_terms.append(PT(c_conj, p_conj))

    if not new_terms:
        return hamiltonian, {
            "energy": None,
            "status": "failed",
            "n_terms_in": len(hamiltonian.terms),
            "n_terms_out": 0,
            "error": "Heff empty after SBRG flow",
        }

    transformed_ham = PauliHamiltonian(
        n_qubits=nq,
        terms=tuple(new_terms),
        metadata={**hamiltonian.metadata, "sbrg_transformed": True},
    )
    ground_energy = min(float(t.val) for t in heff_terms)
    baseline = {
        "energy": ground_energy,
        "status": "ok",
        "n_terms_in": len(hamiltonian.terms),
        "n_terms_out": len(heff_terms),
        "sbrg_version": getattr(_sbrg, "__version__", None),
    }
    return transformed_ham, baseline
