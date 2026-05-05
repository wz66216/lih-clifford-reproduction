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
