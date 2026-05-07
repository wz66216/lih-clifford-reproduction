import json
import importlib.util
from pathlib import Path

import pytest

from lih_repro.chemistry import DependencyUnavailable, load_or_generate_hamiltonian, synthetic_lih_fixture


def test_generate_lih_hamiltonians_defaults_to_repo_data_dir():
    script = _load_script_module()

    root = script.repo_root()

    assert root == Path(__file__).resolve().parents[1]
    assert script.default_output_dir() == root / "data" / "hamiltonians"


def test_generate_lih_hamiltonians_includes_all_config_distances():
    script = _load_script_module()

    distances = script.distances_for_generation()

    assert distances == sorted(distances)
    assert 3.0 in distances
    assert 3.4 in distances


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_lih_hamiltonians.py"
    spec = importlib.util.spec_from_file_location("generate_lih_hamiltonians", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_load_or_generate_rejects_cached_synthetic_fixture_when_disallowed(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "lih_1.400000.json"
    cache_file.write_text(
        json.dumps(synthetic_lih_fixture(1.4).to_dict()),
        encoding="utf-8",
    )

    with pytest.raises(DependencyUnavailable, match="synthetic-fixture.*allow_synthetic_fixture=False"):
        load_or_generate_hamiltonian(1.4, cache_dir=cache_dir, allow_synthetic_fixture=False)


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


def test_dependency_detection_returns_false_when_one_top_level_module_missing(monkeypatch):
    import lih_repro.chemistry as chemistry

    def fake_find_spec(name):
        return object() if name != "openfermionpyscf" else None

    monkeypatch.setattr(chemistry.importlib.util, "find_spec", fake_find_spec)

    assert chemistry._openfermion_available() is False


def test_dependency_detection_returns_true_when_all_top_level_modules_present(monkeypatch):
    import lih_repro.chemistry as chemistry

    monkeypatch.setattr(chemistry.importlib.util, "find_spec", lambda name: object())

    assert chemistry._openfermion_available() is True
