# LiH Zero-Temperature Clifford + kRz Reproduction

This project implements the locally documented algorithmic structure from `Zero and Finite Temperature Quantum Simulations Powered by Quantum Magic` for the LiH zero-temperature `E - E0` curve.

The goal is strict implementation of the described Clifford + `kRz` optimization structure and approximate reproduction of the visual trend in `arXiv-2308.11616v2/figs/ener.pdf`. The project does not claim pointwise reproduction of the paper's hidden numerical data because the local paper files do not provide the exact bond grid, basis, active space, tapering details, SBRG initializer, random seeds, or final optimized circuits.

Install for development:

```powershell
python -m pip install -e ".[dev]"
```

Run the quick smoke experiment:

```powershell
lih-repro --config configs/quick_lih.json
```

If OpenFermion/PySCF are unavailable, the quick configuration may use a deterministic synthetic 8-qubit fixture. Reports generated from that mode are labeled as a software smoke run, not a LiH chemistry reproduction.
