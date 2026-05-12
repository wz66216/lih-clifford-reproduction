# SBRG Integration Design

> **Date:** 2026-05-12
> **Scope:** Add optional SBRG (Spectrum Bifurcation Renormalization Group) baseline energy computation to the LiH Clifford+kRz reproduction pipeline.

---

## Goal

Provide an experimental SBRG energy baseline per bond length. The SBRG baseline is computed **before** the Clifford+kRz optimization, stored as a reference value, and included in the final report. The Clifford+kRz optimization is unchanged.

The integration is **optional**: when SBRG is unavailable or the config flag is off, the pipeline runs exactly as before.

---

## Non-Goals

- Do NOT use SBRG to initialize the Clifford+kRz optimizer (that would require gate-sequence conversion which is out of scope for this iteration).
- Do NOT require SBRG as a hard dependency.
- Do NOT modify the core optimizer or ansatz modules.

---

## Architecture

```
config ("use_sbrg_baseline": true)
    │
    ▼
CLI run_from_config()
    │
    │  Phase 1 (Hamiltonian loading)
    │  for each distance:
    │    ham = load_or_generate_hamiltonian(...)
    │    │
    │    │  if use_sbrg_baseline:
    │    │    try:
    │    │      result = compute_sbrg_baseline(ham)
    │    │       →  sbrg_baselines[distance] = {"energy": ..., "status": "ok" | "unavailable" | "failed"}
    │    │    except SBRGUnavailable:
    │    │       →  sbrg_baselines[distance] = {"status": "unavailable"}
    │    │
    │    hamiltonians[d] = (ham, e0, source)
    │
    │  Phase 2 (optimization) — UNCHANGED
    │
    │  Phase 3 (results aggregation)
    │    results[d][k]["sbrg_energy"] = sbrg_baselines[d].get("energy")
    │
    │  Phase 4 (report)
    │    report includes sbrg_baselines section
```

---

## New Module: `src/lih_repro/sbrg.py`

### Public API

```python
class SBRGUnavailable(RuntimeError):
    """Raised when SBRG is required but not installed."""

def _sbrg_available() -> bool:
    """Check whether SBRG module is importable."""

def pauli_to_sbrg_model(hamiltonian: PauliHamiltonian) -> Any:
    """Convert a PauliHamiltonian to an SBRG Model object.
    Raises SBRGUnavailable if SBRG not installed."""

def compute_sbrg_baseline(hamiltonian: PauliHamiltonian) -> dict:
    """Run SBRG on a PauliHamiltonian and return baseline dict.

    Returns:
        {"energy": float, "status": "ok", "n_terms_in": int, "n_terms_out": int,
         "sbrg_version": str | None}

    Raises SBRGUnavailable if not installed.
    """
```

### SBRG Adapter

The adapter in `pauli_to_sbrg_model` converts our `PauliHamiltonian` into the SBRG library's `Model`/`Term` objects using the `mkMat` Pauli encoding (X=1, Y=2, Z=3, I=0).

The `compute_sbrg_baseline` function:
1. Constructs the SBRG Model
2. Runs `SBRG(model).run()`
3. Extracts the minimum eigenvalue from the effective diagonal Hamiltonian `Heff`
4. Returns the baseline energy and metadata

### Error Handling

- SBRG not installed → `SBRGUnavailable` at import-check time
- SBRG runtime failure → catch exception, return `status: "failed"` with error message

---

## Config Changes

Add to all existing configs (quick_lih.json, fast_curve.json, etc.):

```json
{
  ...existing fields...,
  "use_sbrg_baseline": false
}
```

Default is `false` — opt-in only.

---

## CLI Integration (`src/lih_repro/cli.py`)

`run_from_config()` Phase 1 loop extended:

```python
sbrg_baselines: dict[float, dict] = {}
for distance in config["distances_angstrom"]:
    ...
    if config.get("use_sbrg_baseline", False):
        try:
            result = compute_sbrg_baseline(ham)
            sbrg_baselines[d] = result
        except SBRGUnavailable:
            sbrg_baselines[d] = {"status": "unavailable"}
    ...
```

Phase 3 results inject `sbrg_energy` per distance:

```python
results.append({
    ...
    "sbrg_energy": sbrg_baselines.get(d, {}).get("energy"),
})
```

---

## Results JSON Schema Addition

Each result row gets an optional field:

```json
{
  ...existing fields...,
  "sbrg_energy": -7.85  // optional, null if not computed
}
```

---

## Report Integration (`src/lih_repro/report.py`)

After the existing results table, add an SBRG section:

```
## SBRG Baseline

| distance_angstrom | sbrg_energy | ground_energy | sbrg_gap |
|-------------------|-------------|---------------|----------|
| 1.0               | -7.85       | -7.78         | 0.07     |
...

SBRG energy was computed via `github.com/hongyehu/SBRG` — a classical
real-space RG method that approximately diagonalizes the Pauli Hamiltonian.
```

If SBRG was unavailable, the section shows:

```
## SBRG Baseline (not available)

SBRG was not installed or the `use_sbrg_baseline` flag was off.
```

---

## Testing (`tests/test_sbrg.py`)

**Tests do NOT require the real SBRG library.**

Strategy: test the adapter functions with monkeypatching.

1. **test_sbrg_unavailable** — `_sbrg_available()` returns `False` when module absent
2. **test_pauli_to_sbrg_model_requires_sbrg** — raises `SBRGUnavailable` when not installed
3. **test_compute_sbrg_baseline_unavailable** — raises `SBRGUnavailable` when not installed
4. **test_sbrg_baseline_integration_with_fake** — with monkeypatched `_sbrg_available` and `compute_sbrg_baseline`, verify CLI injects `sbrg_energy` into results
5. **test_cli_sbrg_flag_off_does_nothing** — when `use_sbrg_baseline: false`, results have no `sbrg_energy` field

Test language: ```python
# tests/test_sbrg.py

from lih_repro.sbrg import _sbrg_available, compute_sbrg_baseline, SBRGUnavailable

def test_sbrg_unavailable_when_not_installed():
    """Without the real SBRG module, _sbrg_available() must return False."""
    assert _sbrg_available() is False

def test_compute_sbrg_baseline_raises_when_unavailable():
    """If SBRG is not installed, compute_sbrg_baseline must raise."""
    from lih_repro.pauli import PauliHamiltonian, PauliTerm
    ham = PauliHamiltonian(n_qubits=2, terms=(PauliTerm(1.0, "ZZ"),), metadata={})
    import pytest
    with pytest.raises(SBRGUnavailable):
        compute_sbrg_baseline(ham)
```

---

## File Changes Summary

| File | Action |
|------|--------|
| `src/lih_repro/sbrg.py` | **CREATE** — SBRG adapter module |
| `tests/test_sbrg.py` | **CREATE** — unit/integration tests |
| `src/lih_repro/cli.py` | MODIFY — Phase 1 loop, Phase 3 results |
| `src/lih_repro/report.py` | MODIFY — SBRG section |
| `configs/quick_lih.json` | MODIFY — add `use_sbrg_baseline: false` |
| `configs/fast_curve.json` | MODIFY — add `use_sbrg_baseline: false` |
| `configs/demo_lih.json` | MODIFY — add `use_sbrg_baseline: false` |
| `configs/test_speed.json` | MODIFY — add `use_sbrg_baseline: false` |
| `configs/smoke_real_lih.json` | MODIFY — add `use_sbrg_baseline: false` |

---

## Validation Checklist

- [ ] `python -m pytest -v` → all existing 64 tests pass + new sbrg tests pass
- [ ] `lih-repro --config configs/fast_curve.json` → runs normally with `use_sbrg_baseline: false`
- [ ] With monkeypatched SBRG, results JSON contains `sbrg_energy` per distance
- [ ] Report contains SBRG section when available
- [ ] Report contains "not available" note when unavailable
