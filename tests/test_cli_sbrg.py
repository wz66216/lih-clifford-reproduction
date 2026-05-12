import json

import lih_repro.cli as cli


def test_run_from_config_skips_optional_sbrg_baseline_by_default(monkeypatch, tmp_path):
    called = {"count": 0}

    monkeypatch.setattr(cli, "compute_sbrg_baseline", lambda ham: called.__setitem__("count", called["count"] + 1))
    monkeypatch.setattr(cli, "_sbrg_available", lambda: True)

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

    cli.run_from_config(config_path)

    assert called["count"] == 0


def test_run_from_config_uses_sbrg_initializer(monkeypatch, tmp_path):
    seen = {"count": 0}

    def fake_initializer(ham):
        seen["count"] += 1
        return ham, {"status": "ok", "energy": -1.23}

    monkeypatch.setattr(cli, "compute_sbrg_initializer", fake_initializer)
    monkeypatch.setattr(cli, "_sbrg_available", lambda: True)

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
        "use_sbrg_baseline": True,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    cli.run_from_config(config_path)

    assert seen["count"] == 1
