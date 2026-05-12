import json

import lih_repro.cli as cli

from lih_repro.cli import run_from_config
from lih_repro.chemistry import cache_path_for_distance
from lih_repro.pauli import PauliHamiltonian, PauliTerm


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


def test_run_from_config_records_hartree_fock_comparison_from_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    ham = PauliHamiltonian(
        n_qubits=1,
        terms=(PauliTerm(1.0, "Z"),),
        metadata={"source": "openfermion-pyscf", "hf_energy": 0.25},
    )
    cache_path_for_distance(cache_dir, 1.4).write_text(json.dumps(ham.to_dict()), encoding="utf-8")
    config = {
        "distances_angstrom": [1.4],
        "k_values": [0],
        "n_qubits": 1,
        "layers": 0,
        "seed": 123,
        "continuous_starts": 1,
        "greedy_iterations": 0,
        "n_init": 1,
        "max_workers": 1,
        "output_dir": str(tmp_path / "out"),
        "hamiltonian_cache_dir": str(cache_dir),
        "reference_pdf": str(tmp_path / "ener.pdf"),
        "reference_csv": str(tmp_path / "reference.csv"),
        "allow_synthetic_fixture": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    outputs = run_from_config(config_path)

    rows = json.loads(outputs["results_json"].read_text(encoding="utf-8"))
    assert rows[0]["ground_energy"] == -1.0
    assert rows[0]["hartree_fock_energy"] == 0.25
    assert rows[0]["hartree_fock_gap"] == 1.25
    report_text = outputs["report_md"].read_text(encoding="utf-8")
    assert "HF - E0" in report_text
    assert "0.25" in report_text


def test_safe_worker_count_defaults_to_task_count_and_is_at_least_one(monkeypatch):
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(cli.sys, "platform", "linux")

    assert cli._safe_worker_count(128, 1) == 1
    assert cli._safe_worker_count(128, 8) == 8


def test_safe_worker_count_clamps_windows_limit(monkeypatch):
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(cli.sys, "platform", "win32")

    assert cli._safe_worker_count(128, 128) == 61
    assert cli._safe_worker_count(1000, 70) == 61


def test_run_from_config_clamps_large_max_workers(monkeypatch, tmp_path):
    captured = {}

    class DummyExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            class DummyFuture:
                def result(self_inner):
                    return fn(*args, **kwargs)

            return DummyFuture()

    monkeypatch.setattr(cli.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(cli, "as_completed", lambda futures: list(futures))

    config = {
        "distances_angstrom": [1.4],
        "k_values": [0],
        "n_qubits": 8,
        "layers": 1,
        "seed": 123,
        "continuous_starts": 1,
        "greedy_iterations": 1,
        "max_workers": 128,
        "output_dir": str(tmp_path / "out"),
        "hamiltonian_cache_dir": str(tmp_path / "cache"),
        "reference_pdf": str(tmp_path / "ener.pdf"),
        "reference_csv": str(tmp_path / "reference.csv"),
        "allow_synthetic_fixture": True,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run_from_config(config_path)

    assert captured["max_workers"] == 1


def test_run_from_config_counts_n_init_tasks_for_worker_clamp(monkeypatch, tmp_path):
    captured = {}

    class DummyExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            class DummyFuture:
                def result(self_inner):
                    return fn(*args, **kwargs)

            return DummyFuture()

    monkeypatch.setattr(cli.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(cli, "as_completed", lambda futures: list(futures))

    config = {
        "distances_angstrom": [1.4],
        "k_values": [0],
        "n_qubits": 8,
        "layers": 1,
        "seed": 123,
        "continuous_starts": 1,
        "greedy_iterations": 1,
        "n_init": 4,
        "max_workers": 128,
        "output_dir": str(tmp_path / "out"),
        "hamiltonian_cache_dir": str(tmp_path / "cache"),
        "reference_pdf": str(tmp_path / "ener.pdf"),
        "reference_csv": str(tmp_path / "reference.csv"),
        "allow_synthetic_fixture": True,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run_from_config(config_path)

    assert captured["max_workers"] == 4


def test_run_from_config_rejects_qubit_mismatch(tmp_path):
    config = {
        "distances_angstrom": [1.4],
        "k_values": [0],
        "n_qubits": 7,
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

    try:
        run_from_config(config_path)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "n_qubits=7" in str(exc)
        assert "n_qubits=8" in str(exc)
