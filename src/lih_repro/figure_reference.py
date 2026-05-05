from __future__ import annotations

import csv
from pathlib import Path


def reference_pdf_status(path: Path) -> dict[str, object]:
    pdf_path = Path(path)
    return {
        "path": str(pdf_path),
        "exists": pdf_path.exists(),
        "size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "role": "auxiliary visual validation only",
    }


def load_reference_csv(path: Path) -> dict[str, list[tuple[float, float]]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    curves: dict[str, list[tuple[float, float]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            curve = row["curve"]
            point = (float(row["bond_length"]), float(row["energy_gap"]))
            curves.setdefault(curve, []).append(point)
    return curves
