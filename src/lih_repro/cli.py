from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lih_repro.chemistry import load_or_generate_hamiltonian
from lih_repro.figure_reference import load_reference_csv, reference_pdf_status
from lih_repro.optimizer import OptimizerConfig, optimize_for_k
from lih_repro.plotting import plot_energy_gaps
from lih_repro.report import write_report


def run_from_config(config_path: Path) -> dict[str, Path]:
    config_path = Path(config_path)
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config["hamiltonian_cache_dir"])
    opt_config = OptimizerConfig(
        seed=int(config["seed"]),
        continuous_starts=int(config["continuous_starts"]),
        greedy_iterations=int(config["greedy_iterations"]),
    )
    results: list[dict[str, Any]] = []
    used_synthetic_fixture = False

    for distance in config["distances_angstrom"]:
        hamiltonian = load_or_generate_hamiltonian(
            float(distance),
            cache_dir=cache_dir,
            allow_synthetic_fixture=bool(config["allow_synthetic_fixture"]),
        )
        configured_n_qubits = int(config["n_qubits"])
        actual_n_qubits = int(hamiltonian.n_qubits)
        if actual_n_qubits != configured_n_qubits:
            raise ValueError(
                f"Config n_qubits={configured_n_qubits} does not match cached/generated Hamiltonian n_qubits={actual_n_qubits} "
                f"for distance {float(distance)} in cache_dir={cache_dir}."
            )
        source = str(hamiltonian.metadata.get("source", "unknown"))
        if source == "synthetic-fixture":
            used_synthetic_fixture = True
        ground_energy = hamiltonian.ground_energy()
        for k in config["k_values"]:
            result = optimize_for_k(
                hamiltonian,
                k=int(k),
                layers=int(config["layers"]),
                config=opt_config,
            )
            rows = {
                "distance_angstrom": float(distance),
                "k": int(k),
                "ground_energy": ground_energy,
                "energy": result.energy,
                "energy_gap": result.energy - ground_energy,
                "source": source,
                "theta": list(result.theta),
                "circuit": result.circuit,
                "trace": list(result.trace),
            }
            results.append(rows)

    results_json = output_dir / "results.json"
    plot_png = output_dir / "energy_gap.png"
    report_md = output_dir / "report.md"
    results_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    reference_curves = load_reference_csv(Path(config["reference_csv"]))
    reference_status = reference_pdf_status(Path(config["reference_pdf"]))
    plot_energy_gaps(results, reference_curves=reference_curves, output_path=plot_png)
    write_report(
        output_path=report_md,
        config=config,
        results=results,
        reference_status=reference_status,
        used_synthetic_fixture=used_synthetic_fixture,
    )
    return {"results_json": results_json, "plot_png": plot_png, "report_md": report_md}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LiH Clifford+kRz reproduction experiment")
    parser.add_argument("--config", required=True, help="Path to JSON configuration")
    args = parser.parse_args(argv)
    outputs = run_from_config(Path(args.config))
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
