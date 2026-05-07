from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from lih_repro.pauli import PauliHamiltonian, PauliTerm


class DependencyUnavailable(RuntimeError):
    """Raised when optional chemistry dependencies are required but absent."""


_REQUIRED_CHEMISTRY_MODULES = ("openfermion", "openfermionpyscf", "pyscf")


def cache_path_for_distance(cache_dir: Path, distance_angstrom: float) -> Path:
    return Path(cache_dir) / f"lih_{distance_angstrom:.6f}.json"


def _openfermion_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in _REQUIRED_CHEMISTRY_MODULES)


def load_or_generate_hamiltonian(
    distance_angstrom: float,
    cache_dir: Path,
    allow_synthetic_fixture: bool,
) -> PauliHamiltonian:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path_for_distance(cache_dir, distance_angstrom)
    if path.exists():
        cached = PauliHamiltonian.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if cached.metadata.get("source") == "synthetic-fixture" and not allow_synthetic_fixture:
            raise DependencyUnavailable(
                f"Cached synthetic-fixture Hamiltonian at {path} was found, but allow_synthetic_fixture=False. "
                "Delete the cache or enable synthetic fixtures; real cached Hamiltonians are still allowed."
            )
        return cached

    if _openfermion_available():
        try:
            ham = generate_with_openfermion(distance_angstrom)
        except Exception:
            if not allow_synthetic_fixture:
                raise
            ham = synthetic_lih_fixture(distance_angstrom)
        path.write_text(json.dumps(ham.to_dict(), indent=2), encoding="utf-8")
        return ham

    if allow_synthetic_fixture:
        ham = synthetic_lih_fixture(distance_angstrom)
        path.write_text(json.dumps(ham.to_dict(), indent=2), encoding="utf-8")
        return ham

    raise DependencyUnavailable(
        "OpenFermion/PySCF are unavailable and no cached LiH Hamiltonian exists. "
        "Install with python -m pip install -e \".[chemistry,dev]\" or provide JSON Hamiltonian cache files."
    )


def generate_with_openfermion(distance_angstrom: float) -> PauliHamiltonian:
    """Generate 8-qubit LiH Hamiltonian: STO-3G, frozen Li-1s core, BK + 2-qubit tapering."""
    from openfermion.chem import MolecularData
    from openfermion.transforms import (
        freeze_orbitals,
        get_fermion_operator,
        symmetry_conserving_bravyi_kitaev,
    )
    from openfermion.utils import count_qubits
    from pyscf import gto, scf

    basis = "sto-3g"
    r = float(distance_angstrom)
    mol = gto.M(
        atom=f"Li 0 0 0; H 0 0 {r}",
        basis=basis,
        charge=0,
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    mol_data = MolecularData(
        geometry=[("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, r))],
        basis=basis,
        multiplicity=1,
    )
    mol_data.nuclear_repulsion = mol.energy_nuc()
    mol_data.hf_energy = mf.e_tot
    mol_data.n_orbitals = mol.nao
    mol_data.n_qubits = 2 * mol.nao
    mol_data.n_electrons = mol.nelec[0] + mol.nelec[1]

    mol_data.one_body_integrals = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    two_body_ao = mol.intor("int2e").reshape(4 * [mol.nao])
    # PySCF returns chemist notation (pq|rs); OpenFermion's MolecularData expects
    # h[p,q,r,s] = (ps|qr), which is chemist transposed by (0,2,3,1).
    mol_data.two_body_integrals = np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl",
        two_body_ao,
        mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff,
    ).transpose(0, 2, 3, 1)

    # Freeze Li 1s: spatial orbital 0 = spin-orbitals [0, 1]
    mol_ham = mol_data.get_molecular_hamiltonian()
    fermion_ham = get_fermion_operator(mol_ham)
    fermion_ham = freeze_orbitals(fermion_ham, occupied=[0, 1], unoccupied=[], prune=True)

    n_active_spin_orbitals = 2 * (mol_data.n_orbitals - 1)  # 10 spin-orbitals
    n_active_fermions = mol_data.n_electrons - 2  # 2 active electrons
    qubit_ham = symmetry_conserving_bravyi_kitaev(
        fermion_ham,
        active_orbitals=n_active_spin_orbitals,
        active_fermions=n_active_fermions,
    )

    nq = count_qubits(qubit_ham)
    pauli_terms = []
    for pauli_tuple, coeff in qubit_ham.terms.items():
        chars = ["I"] * nq
        for idx, char in pauli_tuple:
            chars[idx] = char
        pstr = "".join(chars)
        c = float(coeff.real) if hasattr(coeff, "real") else float(coeff)
        if abs(c) > 1e-14:
            pauli_terms.append({"coefficient": c, "pauli": pstr})

    return PauliHamiltonian(
        n_qubits=nq,
        terms=tuple(PauliTerm(t["coefficient"], t["pauli"]) for t in pauli_terms),
        metadata={
            "distance_angstrom": r,
            "source": "openfermion-pyscf",
            "basis": basis,
            "n_electrons": mol_data.n_electrons,
            "n_orbitals": mol_data.n_orbitals,
            "frozen_orbitals": [0, 1],
            "active_spin_orbitals": n_active_spin_orbitals,
            "active_fermions": n_active_fermions,
            "hf_energy": mol_data.hf_energy,
            "nuclear_repulsion": mol_data.nuclear_repulsion,
        },
    )


def synthetic_lih_fixture(distance_angstrom: float) -> PauliHamiltonian:
    r = float(distance_angstrom)
    stretch = r - 1.6
    terms = [
        PauliTerm(-7.85 + 0.18 * stretch * stretch, "IIIIIIII"),
        PauliTerm(0.35 * math.exp(-0.45 * r), "ZIIIIIII"),
        PauliTerm(-0.28 * math.exp(-0.30 * r), "IZIIIIII"),
        PauliTerm(0.22 / (1.0 + r), "IIZIIIII"),
        PauliTerm(-0.18 / (1.0 + 0.5 * r), "IIIZIIII"),
        PauliTerm(0.08 * math.exp(-0.20 * r), "XXXXIIII"),
        PauliTerm(0.05 * math.sin(r), "IIXXYYII"),
        PauliTerm(-0.04 * math.cos(0.5 * r), "IIIIZZII"),
        PauliTerm(0.03 * math.exp(-0.10 * r), "IYIYIIII"),
    ]
    return PauliHamiltonian(
        n_qubits=8,
        terms=tuple(terms),
        metadata={
            "distance_angstrom": r,
            "source": "synthetic-fixture",
            "warning": "This deterministic fixture is for software smoke tests and is not a paper LiH Hamiltonian.",
            "n_qubits": 8,
        },
    )
