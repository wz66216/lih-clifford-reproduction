"""Generate LiH Hamiltonians for the classmate repo cache.

Run in WSL with the repo on PYTHONPATH, for example:
  PYTHONPATH=/mnt/c/Users/15096/Desktop/量子人智大作业opencode/.worktrees/classmate-integration/src \
  python3 /mnt/c/Users/15096/Desktop/量子人智大作业opencode/.worktrees/classmate-integration/scripts/generate_lih_hamiltonians.py
"""
from __future__ import annotations

import json
from pathlib import Path

from lih_repro.chemistry import cache_path_for_distance, generate_with_openfermion

DEFAULT_DISTANCES = [1.0, 1.3, 1.45, 1.6, 1.8, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.5]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_dir() -> Path:
    return repo_root() / "data" / "hamiltonians"


def distances_for_generation() -> list[float]:
    return list(DEFAULT_DISTANCES)


def generate_lih_hamiltonian(distance: float) -> dict:
    """Compute classmate-standard LiH Hamiltonian at the given bond length."""
    ham = generate_with_openfermion(distance)

    return {
        **ham.to_dict(),
    }


def main():
    out_dir = default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in distances_for_generation():
        out_path = cache_path_for_distance(out_dir, d)
        if out_path.exists():
            print(f"SKIP  {d:.1f} Å  (cached)")
            continue
        ham = generate_lih_hamiltonian(d)
        out_path.write_text(json.dumps(ham, indent=2), encoding="utf-8")
        n_terms = len(ham["terms"])
        print(f"OK    {d:.1f} Å  {ham['n_qubits']} qubits  {n_terms} terms")


if __name__ == "__main__":
    main()
