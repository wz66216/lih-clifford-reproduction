# LiH Clifford + kRz Reproduction: 改进指南（供 AI 自动应用）

> 本文档列出了在原同学项目 `Cuuung/LiH_Clifford_Reproduction` 基础上的改进，包括代码位置、改动内容和原因。

---

## 改进 1：Pauli 期望值性能 —— 避免稠密矩阵（最关键）

**文件：** `src/lih_repro/pauli.py`

**原因：** 原版 `PauliHamiltonian.expectation()` 每次计算都重建 4096×4096（8-qubit）的复稠密矩阵（~256 MB），优化器在 L-BFGS-B 里每轮调用数十次，导致计算不可行。

**改动：** 将 `expectation()` 从稠密矩阵乘法改为直接逐项对态矢量的比特翻转 + 相位计算。

**替换 `expectation()` 方法（`src/lih_repro/pauli.py:73-82` 附近）：**

```python
def expectation(self, state: np.ndarray) -> float:
    vector = np.asarray(state, dtype=complex)
    expected_shape = (2**self.n_qubits,)
    if vector.shape != expected_shape:
        raise ValueError(f"state shape {vector.shape} does not match expected {expected_shape}")
    norm = np.linalg.norm(vector)
    if not np.isclose(norm, 1.0):
        raise ValueError(f"state norm must be 1.0, got {norm}")

    indices = np.arange(vector.size)
    value = 0.0 + 0.0j
    for term in self.terms:
        target_indices = indices.copy()
        phases = np.ones(vector.size, dtype=complex)
        for qubit, label in enumerate(term.pauli):
            if label == "I":
                continue

            bit_mask = 1 << (self.n_qubits - 1 - qubit)
            bits = (indices & bit_mask) != 0

            if label == "X":
                target_indices ^= bit_mask
            elif label == "Y":
                target_indices ^= bit_mask
                phases *= np.where(bits, -1j, 1j)
            elif label == "Z":
                phases *= np.where(bits, -1.0, 1.0)

        value += term.coefficient * np.sum(
            np.conjugate(vector[target_indices]) * phases * vector
        )

    return float(value.real)
```

**添加测试（`tests/test_pauli.py`）：**

```python
def test_expectation_matches_dense_matrix_for_mixed_pauli_terms():
    """Verify direct term-wise expectation matches dense matrix expectation."""
    from lih_repro.pauli import PauliHamiltonian, PauliTerm

    terms = (
        PauliTerm(0.5, "XIZ"),
        PauliTerm(-0.3, "IYY"),
        PauliTerm(0.7, "ZZI"),
        PauliTerm(2.0, "III"),
    )
    ham = PauliHamiltonian(n_qubits=3, terms=terms, metadata={})

    rng = np.random.default_rng(42)
    vec = rng.normal(size=8) + 1j * rng.normal(size=8)
    vec = vec / np.linalg.norm(vec)

    direct = ham.expectation(vec)
    dense = np.real(np.conjugate(vec) @ ham.to_dense_matrix() @ vec)
    assert np.isclose(direct, float(dense), atol=1e-12)


def test_expectation_does_not_construct_dense_matrix(monkeypatch):
    """Prove expectation() does not call to_dense_matrix()."""
    from lih_repro.pauli import PauliHamiltonian, PauliTerm

    ham = PauliHamiltonian(
        n_qubits=2, terms=(PauliTerm(1.0, "XX"),), metadata={}
    )

    def _fail(*args, **kwargs):
        raise AssertionError("expectation should evaluate Pauli terms directly")

    monkeypatch.setattr("lih_repro.pauli.PauliHamiltonian.to_dense_matrix", _fail)

    vec = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    result = ham.expectation(vec)
    assert np.isclose(result, 1.0, atol=1e-10)
```

---

## 改进 2：并行任务计数修正

**文件：** `src/lih_repro/cli.py`

**原因：** 原版用 `len(distances) * len(k_values)` 计算 worker 上限，但实际提交的任务数是 `distances × k_values × n_init`。比如 `test_speed.json` 有 400 个实际任务，却只开了 2 个 worker。

**改动：** 将 `_safe_worker_count` 的 `task_count` 参数乘以 `opt_config.n_init`。

**替换（`src/lih_repro/cli.py` 中 `run_from_config` 内，约第 49 行）：**

```python
# 原版（有 bug）
n_task_slots = len(config["distances_angstrom"]) * len(config["k_values"])

# 改为
n_task_slots = len(config["distances_angstrom"]) * len(config["k_values"]) * opt_config.n_init
```

---

## 改进 3：Windows 兼容 —— max_workers 安全限制

**文件：** `src/lih_repro/cli.py`

**原因：** Python `ProcessPoolExecutor` 在 Windows 上 `max_workers` 上限是 61。原版默认 128 在 Windows 直接崩溃。

**添加函数（`src/lih_repro/cli.py` 中，`run_from_config` 之前）：**

```python
def _safe_worker_count(requested_workers: int, task_count: int) -> int:
    requested_workers = int(requested_workers)
    task_count = int(task_count)
    if task_count < 1:
        return 1

    cpu_count = os.cpu_count() or 1
    windows_limit = 61 if sys.platform == "win32" else cpu_count
    safe_workers = min(requested_workers, task_count, cpu_count, windows_limit)
    return max(1, safe_workers)
```

**在 `run_from_config` 中，将：**

```python
n_workers = int(config.get("max_workers", 128))
```

**替换为：**

```python
n_task_slots = len(config["distances_angstrom"]) * len(config["k_values"]) * opt_config.n_init
n_workers = _safe_worker_count(int(config.get("max_workers", 128)), n_task_slots)
```

---

## 改进 4：config n_qubits 与实际哈密顿量校验

**文件：** `src/lih_repro/cli.py`

**原因：** 原版不校验 `config["n_qubits"]` 是否与实际加载的哈密顿量匹配，可能导致错误数据静默产生。

**在 `run_from_config` 的 Phase 1 循环中，加载哈密顿量后添加（约第 64 行）：**

```python
configured_n_qubits = int(config["n_qubits"])
if ham.n_qubits != configured_n_qubits:
    raise ValueError(
        f"Config n_qubits={configured_n_qubits} does not match "
        f"Hamiltonian n_qubits={ham.n_qubits} for distance {d}."
    )
```

---

## 改进 5：哈密顿量缓存生成脚本

**文件：** `scripts/generate_lih_hamiltonians.py`

**原因：** 原版脚本写死了 12-qubit Jordan-Wigner 和特定路径，与同学版 8-qubit Bravyi-Kitaev 标准不一致。

**改动：** 使用同学版 `chemistry.generate_with_openfermion` 和 `cache_path_for_distance`，从所有 `configs/*.json` 自动收集距离参数。

```python
"""Generate 8-qubit LiH Hamiltonians for all config distances."""

import json
import sys
from pathlib import Path

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

sys.path.insert(0, str(repo_root() / "src"))

from lih_repro.chemistry import cache_path_for_distance, generate_with_openfermion

DEFAULT_DISTANCES = [1.0, 1.3, 1.45, 1.6, 1.8, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.5]


def distances_for_generation(configs_dir: Path) -> list[float]:
    """Return sorted union of default distances plus every distance from configs/*.json."""
    distances = set(DEFAULT_DISTANCES)
    configs_dir = Path(configs_dir)
    for config_path in sorted(configs_dir.glob("*.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for d in config.get("distances_angstrom", []):
                distances.add(float(d))
        except Exception as exc:
            raise RuntimeError(f"Failed to parse config at {config_path}: {exc}") from exc
    return sorted(distances)


def main() -> None:
    root = repo_root()
    cache_dir = root / "data" / "hamiltonians"
    distances = distances_for_generation(root / "configs")

    for dist in distances:
        path = cache_path_for_distance(cache_dir, dist)
        if path.exists():
            print(f"skip  {dist:.1f} Å — already cached at {path}")
            continue
        print(f"generating {dist:.1f} Å ...")
        ham = generate_with_openfermion(dist)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ham.to_dict(), indent=2), encoding="utf-8")
        print(f"  -> {path} ({ham.n_qubits} qubits, {len(ham.terms)} terms)")


if __name__ == "__main__":
    main()
```

**运行方式（WSL 内，因为 PySCF 不支持原生 Windows）：**

```bash
wsl bash -lc "cd '/mnt/c/.../repo' && source ~/pyscf-env/bin/activate && PYTHONPATH=src python3 scripts/generate_lih_hamiltonians.py"
```

---

## 改进 6：新增快速验证配置

**文件：** `configs/fast_curve.json`（新建）

**原因：** 原版 `quick_lih.json` 计算量太大（1792 个任务），开发调试时需要一个小型快速配置。

```json
{
  "distances_angstrom": [1.0, 2.0, 3.0, 4.0, 4.5],
  "k_values": [0, 1],
  "n_qubits": 8,
  "layers": 2,
  "seed": 1234,
  "continuous_starts": 2,
  "greedy_iterations": 10,
  "n_init": 1,
  "rz_layer": 1,
  "max_workers": 12,
  "output_dir": "results/fast_curve",
  "hamiltonian_cache_dir": "data/hamiltonians",
  "reference_pdf": "arXiv-2308.11616v2/figs/ener.pdf",
  "reference_csv": "data/reference/ener_digitized.csv",
  "allow_synthetic_fixture": false
}
```

---

## 总结：改进优先级

| 优先级 | 改进 | 影响 |
|--------|------|------|
| **P0** | Pauli 期望值去稠密化 | 性能从"不可行"到"可运行" |
| **P0** | n_init 任务计数修正 | 并行效率从 2 worker → 正常倍数 |
| **P1** | Windows max_workers 安全限制 | 避免 Windows 崩溃 |
| **P1** | n_qubits 校验 | 防止静默错误 |
| **P2** | 缓存生成脚本对齐 | 一键生成所有配置所需的 8-qubit 哈密顿量 |
| **P2** | fast_curve 配置 | 快速开发调试 |
