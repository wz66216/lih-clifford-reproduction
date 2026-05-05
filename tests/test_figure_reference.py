from lih_repro.figure_reference import load_reference_csv, reference_pdf_status


def test_reference_pdf_status_reports_missing_file(tmp_path):
    status = reference_pdf_status(tmp_path / "missing.pdf")

    assert status["exists"] is False
    assert status["path"].endswith("missing.pdf")


def test_load_reference_csv_reads_named_curves(tmp_path):
    csv_path = tmp_path / "ener_digitized.csv"
    csv_path.write_text(
        "curve,bond_length,energy_gap\nHF,1.4,0.5\nk1,1.4,0.2\n",
        encoding="utf-8",
    )

    curves = load_reference_csv(csv_path)

    assert curves["HF"] == [(1.4, 0.5)]
    assert curves["k1"] == [(1.4, 0.2)]
