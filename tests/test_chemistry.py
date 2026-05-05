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


def test_synthetic_fixture_falls_back_when_generation_stub_raises(tmp_path, monkeypatch):
    import lih_repro.chemistry as chemistry

    monkeypatch.setattr(chemistry, "_openfermion_available", lambda: True)
    monkeypatch.setattr(
        chemistry,
        "generate_with_openfermion",
        lambda distance_angstrom: (_ for _ in ()).throw(DependencyUnavailable("stub")),
    )

    ham = load_or_generate_hamiltonian(1.4, cache_dir=tmp_path, allow_synthetic_fixture=True)

    assert ham.metadata["source"] == "synthetic-fixture"
    assert (tmp_path / "lih_1.400000.json").exists()


def test_dependency_detection_ignores_non_import_errors(monkeypatch):
    import lih_repro.chemistry as chemistry

    def boom(name):
        raise RuntimeError("broken install")

    monkeypatch.setattr(chemistry.importlib, "import_module", boom)

    with pytest.raises(RuntimeError, match="broken install"):
        chemistry._openfermion_available()
