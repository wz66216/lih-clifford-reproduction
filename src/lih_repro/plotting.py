from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_energy_gaps(
    rows: Iterable[dict[str, float]],
    reference_curves: dict[str, list[tuple[float, float]]],
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["k"]), []).append((float(row["distance_angstrom"]), float(row["energy_gap"])))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, points in sorted(grouped.items()):
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=f"local k={k}")
    for name, points in sorted(reference_curves.items()):
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], linestyle="--", alpha=0.6, label=f"digitized {name}")
    ax.set_xlabel("LiH bond length (Å)")
    ax.set_ylabel("E - E0")
    ax.set_title("LiH Clifford + kRz reproduction: algorithmic trend, not pointwise paper data")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
