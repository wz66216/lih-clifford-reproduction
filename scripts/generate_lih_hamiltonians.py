"""Generate LiH Hamiltonians with PySCF+OpenFermion in WSL, save as JSON cache.

Run inside WSL: source ~/pyscf-env/bin/activate && python3 /mnt/c/Users/15096/Desktop/量子人智大作业opencode/scripts/generate_lih_hamiltonians.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from openfermion.chem import MolecularData
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.utils import count_qubits
from pyscf import gto, scf

WIN_DATA = Path("/mnt/c/Users/15096/Desktop/量子人智大作业opencode/data/hamiltonians")
# LiH bond lengths from paper figure range (~1.0 to 4.5 Å)
DISTANCES = [1.0, 1.3, 1.45, 1.6, 1.8, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.5]
BASIS = "sto-3g"


def generate_lih_hamiltonian(distance: float) -> dict:
    """Compute STO-3G LiH Hamiltonian at given bond length and return Pauli dict."""
    mol = gto.M(
        atom=f"Li 0 0 0; H 0 0 {distance}",
        basis=BASIS,
        charge=0,
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # Build MolecularData and compute integrals
    mol_data = MolecularData(
        geometry=[("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, distance))],
        basis=BASIS,
        multiplicity=1,
    )
    mol_data.nuclear_repulsion = mol.energy_nuc()
    mol_data.hf_energy = mf.e_tot
    mol_data.n_orbitals = mol.nao
    mol_data.n_qubits = 2 * mol.nao
    mol_data.n_electrons = mol.nelec[0] + mol.nelec[1]

    # One- and two-electron integrals from PySCF
    mol_data.one_body_integrals = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    mol_data.two_body_integrals = mol.intor("int2e").reshape(4 * [mol.nao])
    mol_data.two_body_integrals = np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl",
        mol_data.two_body_integrals,
        mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff,
    )

    # Fermionic → qubit Hamiltonian
    mol_ham = mol_data.get_molecular_hamiltonian()
    fermion_ham = get_fermion_operator(mol_ham)
    qubit_ham = jordan_wigner(fermion_ham)

    nq = count_qubits(qubit_ham)
    pauli_terms = []
    for pauli_tuple, coeff in qubit_ham.terms.items():
        # pauli_tuple = ((idx, 'X'), (idx, 'Y'), ...)
        chars = ["I"] * nq
        for idx, char in pauli_tuple:
            chars[idx] = char
        pstr = "".join(chars)
        c = coeff.real if hasattr(coeff, "real") else float(coeff)
        if abs(c) > 1e-14:
            pauli_terms.append({"coefficient": c, "pauli": pstr})

    # Taper to reduce qubit count (LiH has Z2 symmetry → 2 fewer qubits)
    # Using simple manual taper: freeze first 2 qubits → 10 qubit Hamiltonian
    # Full tapering would reduce to 8 qubits (as in the paper)
    # For now keep all 12 qubits and let the optimizer handle it

    return {
        "n_qubits": nq,
        "terms": pauli_terms,
        "metadata": {
            "distance_angstrom": round(distance, 6),
            "source": "openfermion-pyscf",
            "basis": BASIS,
            "n_electrons": mol_data.n_electrons,
            "n_orbitals": mol_data.n_orbitals,
            "hf_energy": mol_data.hf_energy,
            "nuclear_repulsion": mol_data.nuclear_repulsion,
        },
    }


def main():
    WIN_DATA.mkdir(parents=True, exist_ok=True)
    for d in DISTANCES:
        out_path = WIN_DATA / f"lih_{d:.6f}.json"
        if out_path.exists():
            print(f"SKIP  {d:.1f} Å  (cached)")
            continue
        ham = generate_lih_hamiltonian(d)
        out_path.write_text(json.dumps(ham, indent=2), encoding="utf-8")
        n_terms = len(ham["terms"])
        print(f"OK    {d:.1f} Å  {ham['n_qubits']} qubits  {n_terms} terms")


if __name__ == "__main__":
    main()
