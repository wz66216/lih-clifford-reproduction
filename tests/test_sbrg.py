"""Tests for the SBRG adapter module using monkeypatching.

These tests do NOT require the real SBRG library.
"""

import pytest

from lih_repro.pauli import PauliHamiltonian, PauliTerm
from lih_repro.sbrg import (
    SBRGUnavailable,
    _sbrg_available,
    compute_sbrg_baseline,
    pauli_to_sbrg_model,
)


@pytest.fixture
def two_qubit_ham():
    return PauliHamiltonian(
        n_qubits=2,
        terms=(
            PauliTerm(1.0, "ZZ"),
            PauliTerm(-0.5, "IX"),
            PauliTerm(0.3, "XI"),
        ),
        metadata={},
    )


def test_sbrg_unavailable_when_not_installed(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name, package=None: None)
    assert _sbrg_available() is False


def test_compute_sbrg_baseline_raises_when_unavailable(monkeypatch, two_qubit_ham):
    monkeypatch.setattr("importlib.util.find_spec", lambda name, package=None: None)
    with pytest.raises(SBRGUnavailable, match="SBRG library is not installed"):
        compute_sbrg_baseline(two_qubit_ham)


def test_pauli_to_sbrg_model_raises_when_unavailable(monkeypatch, two_qubit_ham):
    monkeypatch.setattr("importlib.util.find_spec", lambda name, package=None: None)
    with pytest.raises(SBRGUnavailable, match="SBRG library is not installed"):
        pauli_to_sbrg_model(two_qubit_ham)


def test_compute_sbrg_baseline_raises_on_broken_install(monkeypatch, two_qubit_ham):
    """When SBRG is found but import fails, SBRGUnavailable is raised."""
    def fake_find_spec(name, package=None):
        if name == "SBRG":
            return object()  # non-None → "found"
        return None

    def fake_import(name, *args, **kwargs):
        raise ImportError("numba not found")

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(SBRGUnavailable, match="failed to import"):
        compute_sbrg_baseline(two_qubit_ham)


def test_compute_sbrg_baseline_success_with_fake_sbrg(monkeypatch, two_qubit_ham):
    class FakeTerm:
        def __init__(self, val):
            self.val = val

    class FakeSBRG:
        __version__ = "0.1.0"

        class Term:
            def __init__(self, *arg):
                self.mat, self.val = arg if len(arg) == 2 else (arg[0], 1.0)

        @staticmethod
        def mkMat(mu):
            return tuple(mu)

        class Model:
            def __init__(self):
                self.size = 0
                self.terms = []

        class Ham:
            def __init__(self, *arg):
                self.terms = [FakeTerm(-7.5), FakeTerm(-7.2)]

        class SBRG:
            def __init__(self, model):
                self.model = model
                self.Heff = FakeSBRG.Ham()

            def run(self):
                pass

    monkeypatch.setattr("lih_repro.sbrg._import_sbrg", lambda: FakeSBRG)

    result = compute_sbrg_baseline(two_qubit_ham)
    assert result["status"] == "ok"
    assert result["energy"] == pytest.approx(-7.5)
    assert result["n_terms_in"] == 3
    assert result["n_terms_out"] == 2
    assert result["sbrg_version"] == "0.1.0"


def test_compute_sbrg_baseline_failure_with_fake_sbrg(monkeypatch, two_qubit_ham):
    class FakeSBRG:
        __version__ = "0.1.0"

        class Term:
            def __init__(self, *arg):
                self.mat, self.val = arg if len(arg) == 2 else (arg[0], 1.0)

        @staticmethod
        def mkMat(mu):
            return tuple(mu)

        class Model:
            def __init__(self):
                self.size = 0
                self.terms = []

        class SBRG:
            def __init__(self, model):
                self.model = model

            def run(self):
                raise ValueError("SBRG flow diverged")

    monkeypatch.setattr("lih_repro.sbrg._import_sbrg", lambda: FakeSBRG)

    result = compute_sbrg_baseline(two_qubit_ham)
    assert result["status"] == "failed"
    assert "energy" in result
    assert "error" in result
    assert "SBRG flow diverged" in result["error"]
