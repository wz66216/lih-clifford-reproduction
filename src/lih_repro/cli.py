from __future__ import annotations

"""Limit BLAS threads before any numpy/scipy import touches the runtime."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
from pathlib import Path
from typing import Any

from lih_repro.chemistry import load_or_generate_hamiltonian
from lih_repro.figure_reference import load_reference_csv, reference_pdf_status
from lih_repro.optimizer import OptimizerConfig, OptimizationResult, _run_single_restart
from lih_repro.plotting import plot_energy_gaps
from lih_repro.report import write_report


def _safe_worker_count(requested_workers: int, task_count: int) -> int:
    requested_workers = int(requested_workers)
    task_count = int(task_count)
    if task_count < 1:
        return 1

    cpu_count = os.cpu_count() or 1
    windows_limit = 61 if sys.platform == "win32" else cpu_count
    safe_workers = min(requested_workers, task_count, cpu_count, windows_limit)
    return max(1, safe_workers)


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
        n_init=int(config.get("n_init", 1)),
        rz_layer=int(config.get("rz_layer", -1)),
    )
    n_workers = _safe_worker_count(int(config.get("max_workers", 128)), len(config["distances_angstrom"]) * len(config["k_values"]))
    layers = int(config["layers"])
    used_synthetic_fixture = False

    # Phase 1: generate all Hamiltonians sequentially (fast, seconds each)
    print(f"Generating Hamiltonians for {len(config['distances_angstrom'])} bond lengths...")
    hamiltonians: dict[float, tuple] = {}
    for distance in config["distances_angstrom"]:
        d = float(distance)
        ham = load_or_generate_hamiltonian(
            d,
            cache_dir=cache_dir,
            allow_synthetic_fixture=bool(config["allow_synthetic_fixture"]),
        )
        configured_n_qubits = int(config["n_qubits"])
        if ham.n_qubits != configured_n_qubits:
            raise ValueError(
                f"Config n_qubits={configured_n_qubits} does not match "
                f"Hamiltonian n_qubits={ham.n_qubits} for distance {d}."
            )
        if ham.metadata.get("source") == "synthetic-fixture":
            used_synthetic_fixture = True
        e0 = ham.ground_energy()
        hamiltonians[d] = (ham, e0, str(ham.metadata.get("source", "unknown")))
    print(f"  All {len(hamiltonians)} Hamiltonians ready.")

    # Phase 2: build all (distance, k, init_idx) tasks and submit to a single pool
    tasks: list[tuple] = []
    task_keys: list[tuple[float, int]] = []
    for distance in config["distances_angstrom"]:
        d = float(distance)
        ham, _, _ = hamiltonians[d]
        for k in config["k_values"]:
            k_int = int(k)
            for init_idx in range(opt_config.n_init):
                seed = opt_config.seed + 1009 * k_int + 9173 * ham.n_qubits + 7919 * init_idx
                tasks.append((ham, k_int, layers, opt_config, seed))
                task_keys.append((d, k_int))

    print(f"Launching {len(tasks)} optimization tasks across {n_workers} workers...")
    results_map: dict[tuple[float, int], list[OptimizationResult]] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_single_restart, t): i for i, t in enumerate(tasks)}
        done_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            d, k_int = task_keys[idx]
            result = future.result()
            results_map.setdefault((d, k_int), []).append(result)
            done_count += 1
            if done_count % n_workers == 0:
                print(f"  {done_count}/{len(tasks)} tasks completed...")

    # Phase 3: aggregate — take best result per (distance, k)
    results: list[dict[str, Any]] = []
    for distance in config["distances_angstrom"]:
        d = float(distance)
        ham, e0, source = hamiltonians[d]
        for k in config["k_values"]:
            k_int = int(k)
            best = min(results_map[(d, k_int)], key=lambda r: r.energy)
            results.append({
                "distance_angstrom": d,
                "k": k_int,
                "ground_energy": e0,
                "energy": best.energy,
                "energy_gap": best.energy - e0,
                "source": source,
                "theta": list(best.theta),
                "circuit": best.circuit,
                "trace": list(best.trace),
            })

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
    print(f"Done. {len(results)} data points written to {output_dir}")
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
