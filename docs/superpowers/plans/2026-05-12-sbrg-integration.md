# SBRG Baseline Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional SBRG (Spectrum Bifurcation Renormalization Group) baseline energy computation to the LiH reproduction pipeline.

**Architecture:** A new `sbrg.py` adapter module converts `PauliHamiltonian` to SBRG `Model` objects and computes a baseline energy. CLI Phase 1 optionally invokes it; Phase 3 injects `sbrg_energy` into results; the report displays an SBRG section. No core optimizer/ansatz changes. SBRG is optional: when unavailable or config flag is off, the pipeline runs exactly as before.

**Tech Stack:** Python 3.10+, NumPy; optional: hongyehu/SBRG (conda install, NOT a hard dependency)

---

### File Structure

| File | Action | Purpose |
|------|--------|---------|
| `src/lih_repro/sbrg.py` | CREATE | SBRG adapter: availability check, Pauli→Model conversion, baseline compute |
| `tests/test_sbrg.py` | CREATE | Unit tests (monkeypatched, no real SBRG required) |
| `src/lih_repro/cli.py` | MODIFY | Phase 1 SBRG baseline loop, Phase 3 `sbrg_energy` injection |
| `src/lih_repro/report.py` | MODIFY | SBRG baseline section in markdown report |
| `configs/quick_lih.json` | MODIFY | Add `"use_sbrg_baseline": false` |
| `configs/fast_curve.json` | MODIFY | Add `"use_sbrg_baseline": false` |
| `configs/demo_lih.json` | MODIFY | Add `"use_sbrg_baseline": false` |
| `configs/test_speed.json` | MODIFY | Add `"use_sbrg_baseline": false` |
| `configs/smoke_real_lih.json` | MODIFY | Add `"use_sbrg_baseline": false` |

---

### Task 1: Create SBRG Adapter Module

**Files:**
- Create: `src/lih_repro/sbrg.py`

- [ ] **Step 1: Write the module**

Create `src/lih_repro/sbrg.py`:

```python
"""SBRG (Spectrum Bifurcation Renormalization Group) baseline adapter.

The SBRG library (github.com/hongyehu/SBRG) is an *optional* dependency.
When absent, all public functions raise SBRGUnavailable.
When present, this module converts PauliHamiltonian objects into SBRG
Model objects and runs the SBRG flow to produce a baseline energy.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from lih_repro.pauli import PauliHamiltonian


class SBRGUnavailable(RuntimeError):
    """Raised when SBRG is required but the library is not installed."""


_SBRG_SPEC = importlib.util.find_spec("SBRG")


def _sbrg_available() -> bool:
    """Return True if the SBRG library can be imported."""
    return _SBRG_SPEC is not None


# Pauli label → SBRG encoding (X=1, Y=2, Z=3, I=0)
_PAULI_TO_SBRG: dict[str, int] = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def pauli_to_sbrg_model(hamiltonian: PauliHamiltonian) -> Any:
    """Convert a PauliHamiltonian to an SBRG Model object.

    Raises SBRGUnavailable if SBRG is not installed.
    """
    if not _sbrg_available():
        raise SBRGUnavailable(
            "SBRG library is not installed. "
            "Clone github.com/hongyehu/SBRG and install dependencies via conda."
        )

    import SBRG as _sbrg

    terms = []
    for term in hamiltonian.terms:
        mu = [_PAULI_TO_SBRG[label] for label in term.pauli]
        mat = _sbrg.mkMat(mu)
        terms.append(_sbrg.Term(mat, val=float(term.coefficient)))

    return _sbrg.Model(size=hamiltonian.n_qubits, terms=terms)


def compute_sbrg_baseline(hamiltonian: PauliHamiltonian) -> dict[str, Any]:
    """Run SBRG on a PauliHamiltonian and return a baseline energy dict.

    Returns:
        {
            "energy": float,          # SBRG ground-state energy estimate
            "status": "ok",           # "ok" | "failed"
            "n_terms_in": int,        # number of Pauli terms before SBRG
            "n_terms_out": int,       # number of terms in Heff after SBRG
            "sbrg_version": str | None,
        }

    Raises SBRGUnavailable if SBRG is not installed.
    """
    if not _sbrg_available():
        raise SBRGUnavailable(
            "SBRG library is not installed. "
            "Clone github.com/hongyehu/SBRG and install dependencies via conda."
        )

    import SBRG as _sbrg

    model = pauli_to_sbrg_model(hamiltonian)

    try:
        sbrg_instance = _sbrg.SBRG(model)
        sbrg_instance.run()
    except Exception as exc:
        return {
            "energy": None,
            "status": "failed",
            "n_terms_in": len(hamiltonian.terms),
            "n_terms_out": None,
            "sbrg_version": getattr(_sbrg, "__version__", None),
            "error": str(exc),
        }

    # SBRG.Heff is a Ham object; extract the smallest diagonal term
    ground_energy: float | None = None
    n_terms_out = 0
    if hasattr(sbrg_instance, "Heff") and sbrg_instance.Heff is not None:
        heff_terms = sbrg_instance.Heff.terms
        n_terms_out = len(heff_terms) if heff_terms else 0
        if n_terms_out > 0:
            ground_energy = min(float(t.val) for t in heff_terms)

    return {
        "energy": ground_energy,
        "status": "ok",
        "n_terms_in": len(hamiltonian.terms),
        "n_terms_out": n_terms_out,
        "sbrg_version": getattr(_sbrg, "__version__", None),
    }
```

- [ ] **Step 2: Verify module syntax**

Run: `python -c "from lih_repro.sbrg import _sbrg_available, compute_sbrg_baseline, SBRGUnavailable, pauli_to_sbrg_model; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 3: Commit**

```bash
git add src/lih_repro/sbrg.py
git commit -m "feat: add SBRG baseline adapter module"
```

---

### Task 2: Create SBRG Tests

**Files:**
- Create: `tests/test_sbrg.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_sbrg.py`:

```python
"""Tests for the SBRG adapter module.

These tests do NOT require the real SBRG library.
They use monkeypatching to verify integration behavior.
"""

import json

import pytest

from lih_repro.sbrg import (
    SBRGUnavailable,
    _sbrg_available,
    compute_sbrg_baseline,
    pauli_to_sbrg_model,
)
from lih_repro.pauli import PauliHamiltonian, PauliTerm


# ---------------------------------------------------------------------------
# Unit tests — no real SBRG library needed
# ---------------------------------------------------------------------------

@pytest.fixture
def two_qubit_ham():
    return PauliHamiltonian(
        n_qubits=2,
        terms=(
            PauliTerm(1.0, "ZZ"),
            PauliTerm(-0.5, "IX"),
            PauliTerm(0.3, "XI"),
        ),
        metadata={},
    )


def test_sbrg_unavailable_when_not_installed():
    """Without the real SBRG module, _sbrg_available() returns False."""
    assert _sbrg_available() is False


def test_compute_sbrg_baseline_raises_when_unavailable(two_qubit_ham):
    """When SBRG is not installed, compute_sbrg_baseline raises SBRGUnavailable."""
    with pytest.raises(SBRGUnavailable, match="SBRG library is not installed"):
        compute_sbrg_baseline(two_qubit_ham)


def test_pauli_to_sbrg_model_raises_when_unavailable(two_qubit_ham):
    """When SBRG is not installed, pauli_to_sbrg_model raises SBRGUnavailable."""
    with pytest.raises(SBRGUnavailable, match="SBRG library is not installed"):
        pauli_to_sbrg_model(two_qubit_ham)


# ---------------------------------------------------------------------------
# Integration tests — monkeypatched SBRG
# ---------------------------------------------------------------------------

def test_cli_injects_sbrg_energy_when_available(monkeypatch, tmp_path):
    """With monkeypatched SBRG, results JSON contains sbrg_energy per distance."""
    from lih_repro.cli import run_from_config

    # Prepare a minimal config JSON
    config_data = {
        "distances_angstrom": [2.0],
        "k_values": [0],
        "n_qubits": 2,
        "layers": 1,
        "seed": 42,
        "continuous_starts": 1,
        "greedy_iterations": 1,
        "n_init": 1,
        "rz_layer": 1,
        "max_workers": 1,
        "output_dir": str(tmp_path / "out"),
        "hamiltonian_cache_dir": str(tmp_path / "cache"),
        "reference_pdf": "arXiv-2308.11616v2/figs/ener.pdf",
        "reference_csv": "data/reference/ener_digitized.csv",
        "allow_synthetic_fixture": True,
        "use_sbrg_baseline": True,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    # Monkeypatch: pretend compute_sbrg_baseline succeeds
    def fake_compute_sbrg_baseline(ham):
        return {
            "energy": -7.85,
            "status": "ok",
            "n_terms_in": 100,
            "n_terms_out": 20,
            "sbrg_version": "test",
        }

    monkeypatch.setattr(
        "lih_repro.sbrg.compute_sbrg_baseline", fake_compute_sbrg_baseline
    )
    monkeypatch.setattr("lih_repro.sbrg._SBRG_SPEC", object())  # make it truthy

    outputs = run_from_config(config_path)

    results = json.loads(outputs["results_json"].read_text(encoding="utf-8"))
    assert len(results) == 1
    assert results[0]["sbrg_energy"] == -7.85


def test_cli_sbrg_flag_off_leaves_no_sbrg_energy(monkeypatch, tmp_path):
    """When use_sbrg_baseline is false, results rows have no sbrg_energy field."""
    from lih_repro.cli import run_from_config

    config_data = {
        "distances_angstrom": [2.0],
        "k_values": [0],
        "n_qubits": 2,
        "layers": 1,
        "seed": 42,
        "continuous_starts": 1,
        "greedy_iterations": 1,
        "n_init": 1,
        "rz_layer": 1,
        "max_workers": 1,
        "output_dir": str(tmp_path / "out"),
        "hamiltonian_cache_dir": str(tmp_path / "cache"),
        "reference_pdf": "arXiv-2308.11616v2/figs/ener.pdf",
        "reference_csv": "data/reference/ener_digitized.csv",
        "allow_synthetic_fixture": True,
        "use_sbrg_baseline": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    outputs = run_from_config(config_path)

    results = json.loads(outputs["results_json"].read_text(encoding="utf-8"))
    assert len(results) == 1
    assert "sbrg_energy" not in results[0]
```

- [ ] **Step 2: Run tests to verify they fail (import for sbrg module not yet wired)**

Run: `python -m pytest tests/test_sbrg.py::test_sbrg_unavailable_when_not_installed -v`
Since `sbrg.py` was created in Task 1 but `pauli_to_sbrg_model` and `compute_sbrg_baseline` raise when SBRG absent (no CLI changes yet), the unit tests should pass.

Run: `python -m pytest tests/test_sbrg.py::test_sbrg_unavailable_when_not_installed tests/test_sbrg.py::test_compute_sbrg_baseline_raises_when_unavailable tests/test_sbrg.py::test_pauli_to_sbrg_model_raises_when_unavailable -v`
Expected: 3 passed

Run: `python -m pytest tests/test_sbrg.py::test_cli_injects_sbrg_energy_when_available -v`
Expected: FAIL (CLI doesn't yet call `compute_sbrg_baseline`)

Run: `python -m pytest tests/test_sbrg.py::test_cli_sbrg_flag_off_leaves_no_sbrg_energy -v`
Expected: PASS (CLI ignores unknown config key, no sbrg_energy field)

- [ ] **Step 3: Commit**

```bash
git add tests/test_sbrg.py
git commit -m "test: add SBRG adapter and integration tests"
```

---

### Task 3: Integrate SBRG into CLI

**Files:**
- Modify: `src/lih_repro/cli.py` — Phase 1 loop and Phase 3 results

- [ ] **Step 1: Add import for SBRG module**

At the top of `src/lih_repro/cli.py`, add after the existing imports (line 21):

```python
from lih_repro.sbrg import compute_sbrg_baseline, SBRGUnavailable
```

- [ ] **Step 2: Extend Phase 1 loop with SBRG baseline**

In `run_from_config()`, after `used_synthetic_fixture = False` (line 52), add:

```python
sbrg_baselines: dict[float, dict] = {}
use_sbrg = bool(config.get("use_sbrg_baseline", False))
```

In the Phase 1 loop, after `e0 = ham.ground_energy()` (line 72), before `hamiltonians[d] = (ham, e0, ...)` (line 73), add:

```python
if use_sbrg:
    try:
        sbrg_result = compute_sbrg_baseline(ham)
        sbrg_baselines[d] = sbrg_result
        print(f"  SBRG baseline for {d:.1f} Å: {sbrg_result.get('energy')} (status={sbrg_result.get('status')})")
    except SBRGUnavailable:
        sbrg_baselines[d] = {"status": "unavailable"}
        print(f"  SBRG unavailable for {d:.1f} Å")
```

- [ ] **Step 3: Inject sbrg_energy into Phase 3 results**

In the Phase 3 results dict (around line 111-121), add `sbrg_energy` after the `"source"` line:

```python
            results.append({
                "distance_angstrom": d,
                "k": k_int,
                "ground_energy": e0,
                "energy": best.energy,
                "energy_gap": best.energy - e0,
                "source": source,
                "sbrg_energy": sbrg_baselines.get(d, {}).get("energy") if use_sbrg else None,
                "theta": list(best.theta),
                "circuit": best.circuit,
                "trace": list(best.trace),
            })
```

- [ ] **Step 4: Pass SBRG baselines to report**

After the existing `write_report(...)` call (line 131-137), add `sbrg_baselines=sbrg_baselines` to the keyword arguments:

The call should become:

```python
    write_report(
        output_path=report_md,
        config=config,
        results=results,
        reference_status=reference_status,
        used_synthetic_fixture=used_synthetic_fixture,
        sbrg_baselines=sbrg_baselines if use_sbrg else None,
    )
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_sbrg.py -v`
Expected: 5 passed (unit tests + integration tests)

Run: `python -m pytest -v`
Expected: all 69 tests pass

- [ ] **Step 6: Commit**

```bash
git add src/lih_repro/cli.py
git commit -m "feat: integrate SBRG baseline into CLI"
```

---

### Task 4: Add Report Section and Config Fields

**Files:**
- Modify: `src/lih_repro/report.py` — add SBRG section
- Modify: `configs/quick_lih.json` — add `use_sbrg_baseline: false`
- Modify: `configs/fast_curve.json` — add `use_sbrg_baseline: false`
- Modify: `configs/demo_lih.json` — add `use_sbrg_baseline: false`
- Modify: `configs/test_speed.json` — add `use_sbrg_baseline: false`
- Modify: `configs/smoke_real_lih.json` — add `use_sbrg_baseline: false`

- [ ] **Step 1: Update report function signature**

In `src/lih_repro/report.py`, change the function signature (line 7-13):

```python
def write_report(
    output_path: Path,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    reference_status: dict[str, Any],
    used_synthetic_fixture: bool,
    sbrg_baselines: dict[float, dict] | None = None,
) -> None:
```

- [ ] **Step 2: Add SBRG section at end of report**

After the synthetic fixture warning block (after line 61), add:

```python
    # SBRG baseline section
    if sbrg_baselines is not None:
        lines.extend(["", "## SBRG Baseline", ""])
        if any(b.get("status") == "ok" for b in sbrg_baselines.values()):
            lines.append("SBRG energy computed via `github.com/hongyehu/SBRG` — a classical real-space RG method that approximately diagonalizes the Pauli Hamiltonian.")
            lines.append("")
            lines.append("| distance_angstrom | sbrg_energy | ground_energy | sbrg_gap | status |")
            lines.append("|---:|---:|---:|---:|---|")
            for d in sorted(sbrg_baselines):
                b = sbrg_baselines[d]
                energy = b.get("energy")
                sbrg_gap = ""
                if energy is not None:
                    # find matching ground_energy from results
                    e0 = None
                    for row in results:
                        if abs(row["distance_angstrom"] - d) < 1e-6 and row["k"] == 0:
                            e0 = row.get("ground_energy")
                            break
                    if e0 is not None:
                        sbrg_gap = f"{energy - e0:.6f}"
                lines.append(
                    f"| {d} | {energy if energy is not None else ''} | "
                    f"{e0 if e0 is not None else ''} | {sbrg_gap} | {b.get('status', '')} |"
                )
        else:
            lines.append("SBRG baseline was requested but no successful computations were available.")
    elif config.get("use_sbrg_baseline"):
        lines.extend(
            [
                "",
                "## SBRG Baseline (not available)",
                "",
                "SBRG was not installed or the `use_sbrg_baseline` flag was off.",
            ]
        )
```

- [ ] **Step 3: Add use_sbrg_baseline to all config files**

For each of the 5 config files, add `"use_sbrg_baseline": false` before the closing `}`. Example for `configs/fast_curve.json`:

```json
{
  ...existing fields...,
  "allow_synthetic_fixture": false,
  "use_sbrg_baseline": false
}
```

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest -v`
Expected: all 69 tests pass

- [ ] **Step 5: Verify CLI smoke test**

Run: `python -m lih_repro.cli --config configs/smoke_real_lih.json`
Expected: exit 0, results written, no SBRG section (flag is off)

- [ ] **Step 6: Commit**

```bash
git add src/lih_repro/report.py configs/quick_lih.json configs/fast_curve.json configs/demo_lih.json configs/test_speed.json configs/smoke_real_lih.json
git commit -m "feat: add SBRG report section and config fields"
```

---

### Self-Review

**Spec coverage:**
- [x] New `src/lih_repro/sbrg.py` with `SBRGUnavailable`, `_sbrg_available()`, `pauli_to_sbrg_model()`, `compute_sbrg_baseline()` → Task 1
- [x] Pauli→SBRG conversion with X=1,Y=2,Z=3,I=0 encoding → Task 1
- [x] SBRG.run() → Heff energy extraction → Task 1
- [x] SBRGUnavailable on missing library → Task 1
- [x] Test file `tests/test_sbrg.py` → Task 2
- [x] Tests do NOT require real SBRG → Task 2 (all monkeypatched)
- [x] CLI Phase 1 SBRG baseline call → Task 3
- [x] CLI Phase 3 `sbrg_energy` injection → Task 3
- [x] Config field `use_sbrg_baseline: false` on all configs → Task 4
- [x] Report SBRG section → Task 4
- [x] Report "not available" note → Task 4

**Placeholder scan:** No placeholder markers or incomplete sections found.

**Type consistency:** `sbrg_baselines: dict[float, dict]` used consistently across CLI and report. `compute_sbrg_baseline` returns `dict[str, Any]` matching the report consumer. `pauli_to_sbrg_model` converts `PauliHamiltonian` → SBRG `Model` as specified.
