"""Tests for the SBRG adapter module using monkeypatching.

These tests do NOT require the real SBRG library.
"""

import pytest

from lih_repro.pauli import PauliHamiltonian, PauliTerm
from lih_repro.sbrg import (
    SBRGUnavailable,
    _is_identity_pauli,
    _sbrg_available,
    _sbrg_terms,
    _multiply_pauli_strings,
    _pauli_commutes,
    _sbrg_mat_to_pauli_string,
    compute_sbrg_initializer,
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


def test_pauli_commutes():
    assert _pauli_commutes("IZ", "ZI") is True
    assert _pauli_commutes("IX", "XZ") is False
    assert _pauli_commutes("XZ", "ZX") is True
    assert _pauli_commutes("XY", "YX") is True


def test_multiply_pauli_strings():
    r, p = _multiply_pauli_strings("X", "Y")
    assert r == "Z"
    assert abs(p - 1j) < 1e-10
    r, p = _multiply_pauli_strings("X", "X")
    assert r == "I" and abs(p - 1) < 1e-10


def test_sbrg_mat_to_pauli_string():
    class FakeMat:
        def __init__(self, Xs, Zs):
            self.Xs = frozenset(Xs)
            self.Zs = frozenset(Zs)

    mat = FakeMat({0, 2}, {1, 2})
    assert _sbrg_mat_to_pauli_string(mat, 4) == "XZYI"


def test_compute_sbrg_initializer_success(monkeypatch, two_qubit_ham):
    class FakeMat:
        def __init__(self, Xs, Zs):
            self.Xs = frozenset(Xs)
            self.Zs = frozenset(Zs)

    class FakeTerm:
        def __init__(self, *arg):
            if len(arg) >= 2:
                self.mat = arg[0]
                self.val = arg[1]
            elif len(arg) == 1:
                self.mat = arg[0]
                self.val = 1.0
            else:
                self.mat = FakeMat(set(), set())
                self.val = 1.0

    class FakeSBRG:
        __version__ = "0.1.0"
        Term = FakeTerm

        @staticmethod
        def mkMat(mu):
            Xs = {i for i, x in enumerate(mu) if x == 1 or x == 2}
            Zs = {i for i, x in enumerate(mu) if x == 2 or x == 3}
            return FakeMat(Xs, Zs)

        class Model:
            def __init__(self):
                self.size = 0
                self.terms = []

        class Ham:
            def __init__(self, *arg):
                self.terms = [FakeTerm(FakeMat({0}, set()), -5.0)]

        class SBRG:
            def __init__(self, model):
                self.Heff = FakeSBRG.Ham()
                self.RCC = [FakeTerm(FakeMat({0}, set()), 1.0)]

            def run(self):
                pass

    monkeypatch.setattr("lih_repro.sbrg._import_sbrg", lambda: FakeSBRG)

    transformed, baseline = compute_sbrg_initializer(two_qubit_ham)

    assert baseline["status"] == "ok"
    assert baseline["sbrg_version"] == "0.1.0"
    assert len(transformed.terms) > 0


def test_pauli_to_sbrg_model_skips_identity_and_records_offset(monkeypatch):
    class FakeSBRG:
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

    monkeypatch.setattr("lih_repro.sbrg._import_sbrg", lambda: FakeSBRG)
    ham = PauliHamiltonian(
        n_qubits=2,
        terms=(PauliTerm(-3.5, "II"), PauliTerm(0.25, "ZI")),
        metadata={},
    )

    model = pauli_to_sbrg_model(ham)

    assert _is_identity_pauli("II") is True
    assert _is_identity_pauli("ZI") is False
    assert model.identity_offset == pytest.approx(-3.5)
    assert len(model.terms) == 1
    assert model.terms[0].mat == (3, 0)
    assert model.terms[0].val == pytest.approx(0.25)


def test_compute_sbrg_baseline_identity_only_returns_offset(monkeypatch):
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
                self.Heff = []

            def run(self):
                assert self.model.terms == []

            def grndstate_blk(self):
                return None, 0.0

    monkeypatch.setattr("lih_repro.sbrg._import_sbrg", lambda: FakeSBRG)
    ham = PauliHamiltonian(n_qubits=2, terms=(PauliTerm(-4.2, "II"),), metadata={})

    baseline = compute_sbrg_baseline(ham)

    assert baseline["status"] == "ok"
    assert baseline["energy"] == pytest.approx(-4.2)
    assert baseline["n_terms_out"] == 0


def test_sbrg_terms_accepts_plain_list_and_ham_object():
    class FakeTerm:
        pass

    class FakeHam:
        def __init__(self, terms):
            self.terms = terms

    terms = [FakeTerm(), FakeTerm()]

    assert _sbrg_terms(None) == []
    assert _sbrg_terms(terms) == terms
    assert _sbrg_terms(FakeHam(terms)) == terms


def test_compute_sbrg_initializer_preserves_identity_offset_with_list_heff(monkeypatch):
    class FakeMat:
        def __init__(self, Xs, Zs):
            self.Xs = frozenset(Xs)
            self.Zs = frozenset(Zs)

    class FakeTerm:
        def __init__(self, *arg):
            self.mat, self.val = arg if len(arg) == 2 else (arg[0], 1.0)

    class FakeSBRG:
        __version__ = "0.1.0"
        Term = FakeTerm

        @staticmethod
        def mkMat(mu):
            Xs = {i for i, x in enumerate(mu) if x in (1, 2)}
            Zs = {i for i, x in enumerate(mu) if x in (2, 3)}
            return FakeMat(Xs, Zs)

        class Model:
            def __init__(self):
                self.size = 0
                self.terms = []

        class SBRG:
            def __init__(self, model):
                self.model = model
                self.Heff = [FakeTerm(FakeMat({0}, set()), -0.7)]
                self.RCC = []

            def run(self):
                pass

            def grndstate_blk(self):
                return None, -0.7

    monkeypatch.setattr("lih_repro.sbrg._import_sbrg", lambda: FakeSBRG)
    ham = PauliHamiltonian(
        n_qubits=2,
        terms=(PauliTerm(-4.2, "II"), PauliTerm(0.1, "ZI")),
        metadata={},
    )

    transformed, baseline = compute_sbrg_initializer(ham)

    assert baseline["status"] == "ok"
    assert baseline["energy"] == pytest.approx(-4.9)
    assert baseline["n_terms_out"] == 1
    assert transformed.terms[0] == PauliTerm(-4.2, "II")
    assert transformed.terms[1] == PauliTerm(-0.7, "XI")
