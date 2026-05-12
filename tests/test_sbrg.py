import pytest

from lih_repro.pauli import PauliHamiltonian, PauliTerm


def test_sbrg_adapter_raises_when_library_missing():
    from lih_repro.sbrg import SBRGUnavailable, compute_sbrg_baseline, pauli_to_sbrg_model

    ham = PauliHamiltonian(n_qubits=1, terms=(PauliTerm(1.0, "Z"),), metadata={})

    with pytest.raises(SBRGUnavailable):
        pauli_to_sbrg_model(ham)

    with pytest.raises(SBRGUnavailable):
        compute_sbrg_baseline(ham)
