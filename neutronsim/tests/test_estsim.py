"""Simulation for nuclides without evaluated data (physics-law model estimate)."""
import json
import math
import pytest
from neutronsim import cli, core, estsim, nphys
from neutronsim.nuclide import Nuclide
from neutronsim.gui import WizardModel


def nu(t):
    return Nuclide.parse(t)


def test_probabilities_come_from_the_forecast_and_sum_to_one():
    fc = nphys.forecast(nu("Na-23"), 0.0253)
    res = estsim.estimate_result(fc)
    el, cap = ({c.mt: c for c in res.channels}[m] for m in (2, 102))
    assert el.sigma_barn == pytest.approx(next(r.value_b for r in fc.rows if r.name.startswith("elastic")))
    assert cap.sigma_barn == pytest.approx(next(r.value_b for r in fc.rows if r.name.startswith("capture")))
    assert sum(c.probability for c in res.channels) == pytest.approx(1.0) and res.supported
    assert res.channels[0].source is None and "MODEL ESTIMATE" in res.disclaimer


def test_above_20_MeV_still_splits_elastic_and_nonelastic_evenly():
    res = estsim.estimate_result(nphys.forecast(nu("Fe-56"), 30e6))
    assert {c.mt for c in res.channels} == {2, 3}
    assert all(c.probability == pytest.approx(0.5) for c in res.channels)


@pytest.mark.parametrize("e", [1e3 + 1, 1e4, 1e6, 4.9e6, 14e6])
def test_middle_and_fast_energies_use_the_channel_split(e):
    assert estsim.can_simulate(nu("Fe-56"), e) is None
    res = estsim.estimate_result(nphys.forecast(nu("Fe-56"), e))
    mts = {c.mt for c in res.channels}
    assert {2, 102} <= mts and 3 not in mts
    assert sum(c.probability for c in res.channels) == pytest.approx(1.0)
    from neutronsim import viz
    assert all(mt in viz.EMISSIONS for mt in mts if mt > 4 and mt not in (5, 102))       # the visualizer knows every emitted particle
    s = estsim.simulate(nu("Fe-56"), e, 300, 1)[2]
    assert sum(r["count"] for r in s["rows"]) == 300


def test_light_nuclei_refused_at_every_energy():
    for e in (0.0253, 1e4, 1e6):
        with pytest.raises(core.UnsupportedError, match="light|capture"):
            estsim.simulate(nu("H-1"), e, 100, 1)
        assert estsim.can_simulate(nu("H-1"), e)


def test_unpredictable_capture_refused():
    with pytest.raises(core.UnsupportedError, match="capture"):
        estsim.simulate(nu("H-1"), 0.0253, 100, 1)


@pytest.mark.parametrize("e", [0.0253, 1.0, 1e4, 1e6, 4.9e6, 14e6])
def test_every_element_simulates_wherever_allowed(e):
    from neutronsim import explorer as ex
    for z in range(3, 119):
        n = ex.typical_isotope(z)
        if estsim.can_simulate(n, e):
            continue
        fc, res, s = estsim.simulate(n, e, 200, 7)
        assert sum(r["count"] for r in s["rows"]) == 200
        for lo, hi in s["model_probability_range"].values():
            assert 0 <= lo <= hi <= 1 and math.isfinite(lo + hi)


def test_seeded_reproducible_and_matches_probabilities():
    a = estsim.simulate(nu("Na-23"), 0.0253, 20000, 5)[2]
    b = estsim.simulate(nu("Na-23"), 0.0253, 20000, 5)[2]
    assert a == b
    for r in a["rows"]:
        assert abs(r["observed_fraction"] - r["model_probability"]) < 5 * r["sampling_std_error"] + 1e-9
    assert a["model_estimate"] is True


def test_range_brackets_the_central_value_and_closed_shell_is_wide():
    fc = nphys.forecast(nu("Na-23"), 0.0253)
    res = estsim.estimate_result(fc)
    for c in res.channels:
        lo, hi = estsim.prob_ranges(fc)[c.mt]
        assert lo < c.probability < hi
    pb = estsim.prob_ranges(nphys.forecast(nu("Pb-208"), 0.0253))[102]
    assert pb[1] - pb[0] > 0.9                     # honest: the model says almost nothing about a closed shell


def test_sampling_input_still_validated():
    with pytest.raises(core.InputError):
        estsim.simulate(nu("Na-23"), 0.0253, 0, 1)


def test_cli_simulate_is_labelled_and_reproducible(capsys):
    argv = ["simulate", "Na-23", "--energy", "0.0253", "-n", "500", "--seed", "2"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    for s in ("NO EVALUATED DATA", "MODEL ESTIMATE", "plausible range", "do not predict", "not more correct"):
        assert s in out
    cli.main(argv)
    assert capsys.readouterr().out == out
    assert cli.main(argv + ["--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["sample"]["model_estimate"] is True and "model_estimate" in d


def test_cli_simulate_refusals_and_data_isotope_pointer(capsys):
    assert cli.main(["simulate", "Na-23", "--energy", "1", "--unit", "MeV", "-n", "50", "--seed", "1"]) == 0
    assert "MODEL ESTIMATE" in capsys.readouterr().out
    assert cli.main(["simulate", "H-1", "--energy", "1", "--unit", "MeV"]) == 2
    assert "invented" in capsys.readouterr().err
    assert cli.main(["simulate", "nonsense"]) == 2
    capsys.readouterr()
    assert cli.main(["simulate", "U-235", "--energy", "0.0253", "-n", "10"]) == 0
    assert "uses it" in capsys.readouterr().out


def test_xs_and_sample_point_to_simulate_for_isotopes_without_data(capsys):
    for cmd in ("xs", "sample"):
        assert cli.main([cmd, "Na-23", "--energy", "1"]) == 2
        err = capsys.readouterr().err
        assert "unsupported by available data" in err and "neutronsim simulate Na-23" in err


def test_wizard_runs_estimate_draws_and_hands_result_to_visualizer():
    m = WizardModel(acknowledged=True, isotope="Na-23", energy_text="0.0253", unit="eV", mode="sample", n_text="300", seed_text="4")
    out = m.run()
    assert not out.supported and "MODEL ESTIMATE" in out.text and "random draws" in out.text and "NO EVALUATED DATA" in out.text
    assert out.result is not None and out.result.supported
    assert json.loads(out.json_text)["sample"]["n"] == 300
    from neutronsim import viz
    ev = viz.fire(out.result, __import__("random").Random(1))
    assert ev.kind in ("elastic", "capture") and ev.iso == "Na-23"


def test_wizard_draws_at_middle_energy_but_refuses_light_nuclei():
    m = WizardModel(acknowledged=True, isotope="Na-23", energy_text="1", unit="MeV", mode="sample", n_text="100", seed_text="1")
    assert m.check_mode() is None and "random draws" in m.run().text
    m = WizardModel(acknowledged=True, isotope="H-1", energy_text="1", unit="MeV", mode="sample")
    assert "too light" in m.check_mode()
    with pytest.raises(core.InputError):
        m.run()
    m.mode = "xs"
    assert "MODEL ESTIMATE" in m.run().text


def test_all_command_registered_and_steps_are_consistent():
    a = cli.build_parser().parse_args(["all", "-v", "--no-tests"])
    assert a.fn is cli.cmd_all and a.verbose and a.no_tests
    from neutronsim import runall
    assert runall.main(run_tests=False) == 0


def test_gui_flag_and_launchers(capsys, monkeypatch=None):
    a = cli.build_parser().parse_args(["all", "--gui", "--no-tests"])
    assert a.gui
    from neutronsim import runall, gui
    opened = []
    orig, gui.main = gui.main, (lambda: opened.append(1) or 0)
    try:
        assert runall.main(run_tests=False, gui=True) == 0 and opened == [1]
    finally:
        gui.main = orig
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert (root / "run_gui.py").exists() and (root / "run_all.py").exists()


def test_visualizer_result_exists_for_every_mode_and_energy():
    m = WizardModel(acknowledged=True, isotope="Na-23", energy_text="0.0253", unit="eV", mode="xs")
    out = m.run()                                             # cross-sections mode used to give result=None: Visualize stayed locked
    assert out.result is not None and out.result.channels and not out.supported
    m.energy_text, m.unit = "1", "MeV"                        # 1 keV - 20 MeV: now fireable
    out = m.run()
    assert out.result is not None and len(out.result.channels) >= 2 and "no neutron can be fired" not in out.text
    m.isotope = "H-1"                                         # too light: nucleus can be drawn, nothing can be fired
    out = m.run()
    assert out.result is not None and out.result.channels == [] and "no neutron can be fired" in out.text
    assert "too light" in out.result.reason
    m.isotope = "C-14"; m.energy_text, m.unit = "0.0253", "eV"
    assert m.run().result is not None                         # capture unpredictable: still drawable


def test_cli_viz_on_isotope_without_data_uses_the_estimate(capsys):
    from neutronsim import viz
    seen = {}
    orig, viz.main = viz.main, (lambda res: seen.setdefault("res", res) and 0)
    try:
        assert cli.main(["viz", "Na-23", "--energy", "0.0253"]) == 0
        assert "NO EVALUATED DATA" in capsys.readouterr().out and seen["res"].channels
        seen.clear()
        assert cli.main(["viz", "Na-23", "--energy", "1", "--unit", "MeV"]) == 0
        assert seen["res"].channels
        seen.clear()
        assert cli.main(["viz", "H-1", "--energy", "1", "--unit", "MeV"]) == 0
        assert not seen["res"].channels and "too light" in capsys.readouterr().out
    finally:
        viz.main = orig
