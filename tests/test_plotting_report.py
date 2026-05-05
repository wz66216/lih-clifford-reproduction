from lih_repro.plotting import plot_energy_gaps
from lih_repro.report import write_report


def test_plot_energy_gaps_creates_png(tmp_path):
    output = tmp_path / "energy_gap.png"
    rows = [
        {"distance_angstrom": 1.4, "k": 0, "energy_gap": 0.3},
        {"distance_angstrom": 1.4, "k": 1, "energy_gap": 0.1},
        {"distance_angstrom": 2.0, "k": 0, "energy_gap": 0.4},
        {"distance_angstrom": 2.0, "k": 1, "energy_gap": 0.2},
    ]

    plot_energy_gaps(rows, reference_curves={}, output_path=output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_write_report_mentions_reproduction_boundary(tmp_path):
    output = tmp_path / "report.md"
    write_report(
        output_path=output,
        config={"seed": 1234},
        results=[{"distance_angstrom": 1.4, "k": 0, "energy_gap": 0.3}],
        reference_status={"exists": True, "path": "ener.pdf", "role": "auxiliary visual validation only"},
        used_synthetic_fixture=True,
    )

    text = output.read_text(encoding="utf-8")
    assert "not a pointwise reproduction" in text
    assert "synthetic fixture" in text
