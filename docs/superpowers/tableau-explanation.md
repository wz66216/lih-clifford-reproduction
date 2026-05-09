# LiH 论文的 Tableau（广义稳定子表示）详解

> 论文 "Zero and Finite Temperature Quantum Simulations Powered by Quantum Magic" (arXiv:2308.11616v2) 中 Clifford + kRz 电路的核心经典模拟方法。

---

## 1. 为什么需要 Tableau？

Clifford 门 (H、S、CNOT) 组成的电路虽然能产生大量纠缠，但根据 Gottesman-Knill 定理，可以用经典计算机在多项式时间内模拟。Tableau 正是这种高效模拟的核心数据结构。

但论文的电路除了 Clifford 门，还掺杂了少量非 Clifford 门（Rz gate）。称这种电路为 **Clifford + kRz**（k = 非 Clifford 门的个数）。论文通过"广义稳定子表示"，把纯 Clifford 的 tableau 扩展来模拟这些掺杂电路。

---

## 2. 概念体系

```
┌─────────────────────────────────────────────────────────────┐
│                     Full Tableau T                           │
│                                                              │
│  ┌───────────────────────┐  ┌───────────────────────────┐   │
│  │   Stabilizer Group S   │  │   Destabilizer Group D    │   │
│  │   s₁, s₂, ..., sₙ      │  │   d₁, d₂, ..., dₙ         │   │
│  │   彼此交换              │  │   dᵢ 只和 sᵢ 反对易       │   │
│  │   -I ∉ S               │  │   和 sⱼ (j≠i) 交换        │   │
│  └───────────┬───────────┘  └──────────────┬──────────────┘   │
│              │                              │                   │
│       定义稳定子态 |ψ_S⟩              生成稳定子基             │
│       ρ_S = (1/2ⁿ)Π(I+sᵢ)            {d|ψ_S⟩ | d∈D}          │
│                                        (2ⁿ 个正交基矢)          │
│              └──────────────┬───────────────┘                   │
│                             │                                   │
│              任意纯态: ρ = Σᵢⱼ χᵢⱼ dᵢ ρ_S dⱼ                │
│              χ 是 2ⁿ×2ⁿ 系数矩阵（通常稀疏）                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Pauli 群

n-qubit 体系上所有 Pauli 矩阵张量积的集合（带相位 ±1, ±i）：

$$G_n = \{i^k \\;\\sigma^{i_1} \\otimes \\sigma^{i_2} \\otimes \\cdots \\otimes \\sigma^{i_n} \\;|\\; k,i_h \\in \\{0,1,2,3\\}\\}$$

共 4ⁿ⁺¹ 个元素。

### 2.2 稳定子群 S

Pauli 群的子群，满足两个条件：
1. 阿贝尔群（所有元素彼此交换）
2. 不含 −I

n-qubit 的满稳定子群由 n 个生成元张成：S = ⟨s₁, s₂, ..., sₙ⟩，共有 2ⁿ 个元素。

### 2.3 稳定子态

被 S 中所有元素"稳定"的唯一纯态：

$$\rho_S = \frac{1}{2^n}\prod_{i=1}^{n}(I + s_i)$$

**例子 (n=2):**
```
s₁ = X⊗X,  s₂ = Z⊗Z  →  |ψ_S⟩ = (|00⟩ + |11⟩)/√2  (Bell 态)
```

### 2.4 解稳定子群 D

与 S 配对，有 n 个生成元 d₁, ..., dₙ，满足：

```
{dᵢ, sᵢ} = 0   (反对易)
[dᵢ, sⱼ] = 0   (j≠i, 交换)
```

**例子 (对应上面的 Bell 态):**
```
d₁ = Z⊗I,  d₂ = I⊗X
```

### 2.5 稳定子基

$$\mathcal{B}(\mathcal{T}) = \\{ d \\,|\\psi_S\\rangle \\;|\\; d \\in \mathcal{D} \\}$$

有 2ⁿ 个态，构成完整的正交基。本质上是从 |0...0⟩ 旋转到以 tableau 定义的新基底。

### 2.6 广义稳定子表示

任意纯态在这个基底下的展开：

$$\rho = \sum_{i=1}^{2^n} \sum_{j=1}^{2^n} \chi_{ij} \\; d_i \\, \rho_S \\, d_j$$

**χ 的稀疏度 ‖χ‖₀ 决定模拟复杂度。**

---

## 3. 三种门在 Tableau 下的演化

### 3.1 Clifford 门 → O(n²)

作用只改变 tableau，不改变 χ：

$$C \rho C^{\dagger} = \sum_{ij} \chi_{ij} \\; (C d_i C^{\dagger}) \\; (C \rho_S C^{\dagger}) \\; (C d_j C^{\dagger})$$

等价于：

$$(\chi, \mathcal{S}, \mathcal{D}) \rightarrow (\chi, \mathcal{S}', \mathcal{D}')$$

**每个 Clifford 门只需 O(n²) 更新 S 和 D。** 这正是 Gottesman-Knill 定理的核心。

### 3.2 非 Clifford 门 (Rz) → Pauli Channel 展开

Rz(θ) 不是 Clifford 门。把它写成 Pauli 串的线性组合：

$$R_z(\theta) = \cos\frac{\theta}{2} \\; I - i \sin\frac{\theta}{2} \\; Z$$

$$\Rightarrow R_z(\theta) \\, \rho \\, R_z^{\dagger}(\theta) = \sum_{m,n} \phi_{mn} \\; P_m \\, \rho \\, P_n^{\dagger}$$

其中 Pₘ, Pₙ 是 Pauli 串，φₘₙ 是复系数。**一个 Rz 门有 4 项。**

### 3.3 Pauli Channel 演化 → 更新 χ

对 Pauli channel 中的每一项 Pₘ ρ Pₙ†：

1. 把 Pₘ, Pₙ 用 tableau 分解（见第 4 节）
2. 更新 χ 矩阵：χ 中原来在位置 (i, j) 的项迁移到新位置 (i', j')
3. 对所有项求和

$$\chi'_{i'j'} = \sum_{m,n} \phi_{mn} \\, \chi_{ij} \\, \alpha_m \alpha_n^* \\, (-1)^{c_m \cdot i + c_n \cdot j}$$

---

## 4. Pauli 分解算法（Algorithm 1 —— 核心操作）

**目的：** 给定任意 Pauli 串 P，在 tableau T = (S, D) 下分解为：

$$P = \alpha \cdot d_{b} \cdot s_{c}$$

其中 α 是相位，b 和 c 是 n 位二元向量，分别标记参与乘积的 destabilizer generator 和 stabilizer generator。

```
Algorithm: Decompose(P, T)

  Initialize: b = (0,...,0), c = (0,...,0), phase = 0
  
  Step 1 — 通过 destabilizer 消除 P 与 S 的反对易
  for i = 0 to n-1:                    // 遍历 stabilizer generators
    if {P, s_i} = 0:                   // P 和 s_i 反对易
      b_i = 1                          // 标记 d_i 参与
      phase -= ipow(P, d_i)            // 更新相位
      P = P * d_i                      // 乘以 d_i 消除反对易
  
  Step 2 — 通过 stabilizer 消除 P 与 D 的反对易
  for i = 0 to n-1:                    // 遍历 destabilizer generators
    if {P, d_i} = 0:                   // P 和 d_i 反对易
      c_i = 1                          // 标记 s_i 参与
      phase -= ipow(P, s_i)            // 更新相位
      P = P * s_i                      // 乘以 s_i 消除反对易
  
  Return: α = exp(iπ·phase/2), b, c
```

**关键直觉：**
- Step 1：P 和某个 sᵢ 反对易时，乘上对应的 dᵢ（dᵢ 与 sᵢ 反对易），使得乘积和 sᵢ 交换。
- Step 2：类似地处理剩余反对易关系。
- 最终 P 和所有 generator 都交换，必是 identity，仅剩相位。

---

## 5. 复杂度分析

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| 1 个 Clifford 门 | **O(n²)** | 更新 tableau (S,D) |
| L 层 Clifford | **O(n³ · L)** | 每层有 n 个两比特门 |
| 1 个 Rz 门 | **O(‖χ‖₀)** | 4 项 Pauli channel |
| k 个 Rz 门 | **O(4ᵏ ‖χ‖₀)** | 每加一个 Rz，χ 稀疏度 ×4 |
| **总模拟** | **O(n³L + 4ᵏ ‖χ‖₀)** | |

**关键洞察：**

$$\text{当 } k \leq O(\log n):\; 4^k = \text{poly}(n) \Rightarrow \text{整个模拟是多项式时间}$$

这称为"量子魔法梯子"(Quantum Magic Ladder)：k 从 0 开始逐步增加，每增加 1 就"多爬一级"，精度提高但经典计算量翻 4 倍。

---

## 6. 与全态矢量模拟的对比

| | 论文 (Tableau) | 当前实现 (State Vector) |
|---|---|---|
| 数据存储 | 2ⁿ×2ⁿ 稀疏矩阵 χ | 2ⁿ 复向量 |
| Clifford 门代价 | O(n²) | O(2ⁿ) |
| Rz 门代价 | 4ᵏ · ‖χ‖₀ | O(2ⁿ) |
| n=8, k=2, ‖χ‖₀~100 | ~1600 操作 | 256 振幅 |
| n=20, k=2 | 可模拟 | **不可行** |
| n=8 的精度 | 完全相同 | 完全相同 |

**结论：对于 n=8 的 LiH，两种方法计算结果应一致。** 当前实现的能量差偏高（10⁻¹~10⁰ vs 论文 10⁻⁴~10⁻¹），根因是优化预算不够，不是模拟方法差异。

---

## 7. 形象总结

可以把 Tableau 想象为"计算坐标系"：

```
全态矢量模拟 (= 笛卡尔坐标):
  每个态存 2ⁿ 个复数 → 8 qubit = 256 个复数 → 直接操作

Tableau 模拟 (= 极坐标):
  利用 Clifford 结构 → 大部分操作只更新"角度"（S,D）
  非 Clifford 才更新"径向"（χ）→ 稀疏 → 高效
```

论文利用这个"极坐标"优势，在 n 较大时仍能模拟 k 较小的 Clifford+kRz 电路。

---

> 参考：arXiv:2308.11616v2, Appendix A "Classical simulation with the generalized stabilizer representation"
