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


def test_plot_energy_gaps_draws_hartree_fock_curve(monkeypatch, tmp_path):
    calls = []

    class FakeAxes:
        def plot(self, xs, ys, **kwargs):
            calls.append((list(xs), list(ys), kwargs))

        def set_xlabel(self, value):
            pass

        def set_ylabel(self, value):
            pass

        def set_title(self, value):
            pass

        def grid(self, *args, **kwargs):
            pass

        def legend(self, *args, **kwargs):
            pass

    class FakeFigure:
        def tight_layout(self):
            pass

        def savefig(self, output_path, dpi):
            output_path.write_bytes(b"png")

    fake_fig = FakeFigure()
    monkeypatch.setattr("lih_repro.plotting.plt.subplots", lambda figsize: (fake_fig, FakeAxes()))
    monkeypatch.setattr("lih_repro.plotting.plt.close", lambda fig: None)
    output = tmp_path / "energy_gap.png"
    rows = [
        {"distance_angstrom": 1.0, "k": 0, "energy_gap": 0.3, "hartree_fock_gap": 0.8},
        {"distance_angstrom": 2.0, "k": 0, "energy_gap": 0.2, "hartree_fock_gap": 0.9},
    ]

    plot_energy_gaps(rows, reference_curves={}, output_path=output)

    hf_calls = [kwargs for _, _, kwargs in calls if kwargs.get("label") == "Hartree-Fock"]
    assert hf_calls
    assert hf_calls[0]["linestyle"] == "--"
    assert hf_calls[0]["color"] == "black"


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


def test_write_report_includes_hartree_fock_columns(tmp_path):
    output = tmp_path / "report.md"
    write_report(
        output_path=output,
        config={"seed": 1234},
        results=[
            {
                "distance_angstrom": 1.4,
                "k": 0,
                "ground_energy": -1.0,
                "energy": -0.7,
                "energy_gap": 0.3,
                "hartree_fock_energy": 0.25,
                "hartree_fock_gap": 1.25,
            }
        ],
        reference_status={"exists": True, "path": "ener.pdf", "role": "auxiliary visual validation only"},
        used_synthetic_fixture=False,
    )

    text = output.read_text(encoding="utf-8")
    assert "HF E" in text
    assert "HF - E0" in text
    assert "1.25" in text
