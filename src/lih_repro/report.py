from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(
    output_path: Path,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    reference_status: dict[str, Any],
    used_synthetic_fixture: bool,
    sbrg_baselines: dict[float, dict[str, object]] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LiH Zero-Temperature Reproduction Report",
        "",
        "This report summarizes a strict implementation of the locally documented Clifford + kRz algorithm structure.",
        "It is not a pointwise reproduction of the paper's hidden numerical data.",
        "",
        "## Boundary",
        "",
        "The local paper files omit the exact bond grid, basis, active space, tapering details, SBRG initialization, seeds, and final optimized circuits.",
        "The PDF figure is used as auxiliary visual validation only.",
        "",
        "## Configuration",
        "",
        f"- Seed: {config.get('seed')}",
        f"- Distances: {config.get('distances_angstrom')}",
        f"- k values: {config.get('k_values')}",
        f"- Synthetic fixture used: {used_synthetic_fixture}",
        "",
        "## Reference Figure Status",
        "",
        f"- Path: {reference_status.get('path')}",
        f"- Exists: {reference_status.get('exists')}",
        f"- Role: {reference_status.get('role')}",
        "",
        "## Results",
        "",
        "| distance_angstrom | k | E0 | E | E - E0 | SBRG E | SBRG status | source |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        ground_energy = row.get("ground_energy")
        energy = row.get("energy")
        source = row.get("source", "")
        lines.append(
            f"| {row['distance_angstrom']} | {row['k']} | {ground_energy if ground_energy is not None else ''} | "
            f"{energy if energy is not None else ''} | {row.get('energy_gap', '')} | {row.get('sbrg_energy', '')} | "
            f"{row.get('sbrg_status', '')} | {source} |"
        )
    if used_synthetic_fixture:
        lines.extend(
            [
                "",
                "## Synthetic Fixture Warning",
                "",
                "At least one Hamiltonian came from the deterministic synthetic fixture. This validates the software pipeline but is not a LiH chemistry reproduction.",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
