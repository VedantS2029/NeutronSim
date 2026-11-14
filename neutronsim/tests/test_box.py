"""Box test (step B1): bookkeeping, physics sanity checks, scope refusals, input validation."""
import json

import pytest

from neutronsim import box, cli, core


def run(iso, e, **kw):
    kw.setdefault("n", 1500)
    kw.setdefault("seed", 7)
    return box.run_box(iso, e, **kw)


@pytest.mark.parametrize("iso,e,f", [("B-10", 0.0253, 1.0), ("B-10", 14e6, 1.0), ("Al-27", 14e6, 1.0),
                                     ("Ag-109", 1e6, 1.0), ("Ag-109", 14e6, 0.5)])
def test_energy_and_neutron_bookkeeping(iso, e, f):
    r = run(iso, e, density_factor=f, n=800)
    assert r.absorbed + r.leaked + r.alive_end == r.n_total            # every neutron is accounted for
    assert abs(r.closure_MeV) < 1e-6 * max(1.0, abs(r.energy_in_MeV))  # source + Q = heat + leaked + alive
    assert r.alive_t[-1] == r.alive_end and r.absorbed_t[-1] == r.absorbed and r.leaked_t[-1] == r.leaked
    assert abs(r.heat_MeV_t[-1] - r.heat_MeV) < 1e-9 * max(1.0, abs(r.heat_MeV))
    assert sum(r.tally.values()) == r.collisions


def test_b10_thermal_heat_per_absorption_is_the_alpha_q_value():
    r = run("B-10", 0.0253, density_factor=1.0)
    assert r.absorbed == r.n_source and r.leaked == 0
    assert r.heat_per_source_MeV == pytest.approx(2.79, rel=0.02)       # (n,alpha) Q


def test_al_capture_heat_is_the_stored_q():
    r = run("Al-27", 0.0253, density_factor=1.0, side_cm=100.0, n=500)
    assert r.absorbed > 0
    cap = r.tally["MT102 (n,gamma) radiative capture"]
    assert cap == r.absorbed
    assert r.heat_MeV / r.absorbed > 7.7                                 # 7.725 MeV + the neutron's energy


def test_density_slider_effect_b10():
    thin = run("B-10", 0.0253, density_factor=1e-4)
    thick = run("B-10", 0.0253, density_factor=1.0)
    assert thin.leaked > 0.5 * thin.n_source          # nearly transparent at 1e-4 of solid
    assert thick.leaked == 0
    assert thin.absorbed < thick.absorbed
    assert thin.heat_MeV < thick.heat_MeV


def test_vacuum_like_density_everything_leaks_without_collisions():
    r = run("Al-27", 1e6, density_factor=1e-9)
    assert r.leaked == r.n_source and r.collisions <= 2 and r.heat_MeV < 1e-3 * r.n_source


def test_seed_reproducible_and_different_seed_differs():
    a, b, c = run("Ag-109", 1e6, seed=3), run("Ag-109", 1e6, seed=3), run("Ag-109", 1e6, seed=4)
    assert (a.absorbed, a.leaked, a.heat_MeV) == (b.absorbed, b.leaked, b.heat_MeV)
    assert (a.absorbed, a.leaked, a.heat_MeV) != (c.absorbed, c.leaked, c.heat_MeV)


def test_time_limit_leaves_neutrons_alive():
    r = run("Al-27", 0.0253, density_factor=1.0, side_cm=100.0, t_end_s=1e-6, n=300)
    assert r.alive_end > 0 and r.absorbed + r.leaked + r.alive_end == r.n_total


def test_n2n_births_only_where_data_allow():
    low = run("Ag-109", 1e6, n=400)
    high = run("Ag-109", 14e6, n=400)
    assert low.n_born == 0 and high.n_born > 0


def test_fissile_and_actinide_isotopes_are_refused():
    with pytest.raises(core.UnsupportedError, match="fission"):
        box.run_box("U-235", 1e6)
    with pytest.raises(core.UnsupportedError, match="actinide"):
        box.run_box("U-238", 1e6)
    with pytest.raises(core.UnsupportedError, match="reference density"):
        box.run_box("At-210", 1.0)
    assert "U-235" not in box.boxable_isotopes()
    assert set(box.boxable_isotopes()) == {"Ag-109", "Al-27", "B-10"}
    assert cli.main(["box", "U-235", "--energy", "1", "--unit", "MeV"]) == 2


@pytest.mark.parametrize("kw", [dict(n=0), dict(n=box.MAX_NEUTRONS + 1), dict(side_cm=0), dict(side_cm=1e6),
                                dict(density_factor=0), dict(density_factor=-1), dict(density_factor=box.MAX_DENSITY_FACTOR + 0.1),
                                dict(t_end_s=0), dict(source="corner"), dict(density_g_cm3=1.0, density_factor=1.0)])
def test_input_validation(kw):
    with pytest.raises(core.InputError):
        box.run_box("B-10", 0.0253, **kw)


def test_energy_outside_data_range_refused():
    with pytest.raises(core.UnsupportedError):
        box.run_box("B-10", 1e9)


def test_q_table_has_a_source_for_every_channel():
    r = run("Al-27", 14e6, n=50)
    assert r.q_table and all(src for *_, src in r.q_table)
    q = {mt: (v, s) for mt, _, v, s in r.q_table}
    assert q[102] == (pytest.approx(7.7252, abs=1e-3), "evaluation (stored)")
    assert q[107][1].startswith("implied by the threshold")


def test_cli_text_and_json(capsys):
    assert cli.main(["box", "B-10", "--energy", "0.0253", "-n", "200", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "BOX TEST" in out and "heat deposited" in out and "Fission is out of scope" in out
    assert cli.main(["box", "Al-27", "--energy", "1", "--unit", "MeV", "-n", "100", "--seed", "1", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["n_source"] == 100 and "heat_MeV" in d


# ---- any isotope: model-estimate box, and trace recording for the animation
def test_model_isotope_box_runs_and_conserves_energy():
    r = box.run_box("Fe-56", 14e6, density_factor=0.1, n=200, seed=3)
    assert r.model_estimate and r.n_source == 200
    assert r.absorbed + r.leaked + r.alive_end == r.n_total
    assert abs(r.closure_MeV) < 1e-6
    assert any("MODEL ESTIMATE" in a for a in r.assumptions) and "MODEL ESTIMATE" in box.render(r)
    assert r.reference_density_g_cm3 == pytest.approx(7.874)


def test_model_isotope_density_slider_changes_the_physics():
    thin = box.run_box("Na-23", 1e6, density_factor=1e-4, n=300, seed=1)
    thick = box.run_box("Na-23", 1e6, density_factor=1.0, n=300, seed=1)
    assert thin.collisions < thick.collisions and thin.leaked >= thick.leaked


def test_model_box_refuses_where_the_model_cannot_split_outcomes():
    with pytest.raises(core.UnsupportedError, match="capture"):
        box.run_box("H-1", 0.0253)


def test_data_isotopes_still_use_data_not_the_model():
    r = box.run_box("B-10", 0.0253, density_factor=0.01, n=100, seed=1)
    assert not r.model_estimate and r.heat_per_source_MeV > 0


def test_tracing_does_not_change_the_physics():
    a = box.run_box("B-10", 0.0253, density_factor=0.01, n=200, seed=5)
    b = box.run_box("B-10", 0.0253, density_factor=0.01, n=200, seed=5, trace=40)
    key = lambda r: (r.absorbed, r.leaked, r.alive_end, r.collisions, r.heat_MeV, r.n_born)
    assert key(a) == key(b) and len(b.traces) == 40 and not a.traces


def test_traces_are_consistent_with_the_run():
    r = box.run_box("Fe-56", 14e6, density_factor=0.3, n=60, seed=2, trace=60)
    h = r.side_cm / 2 + 1e-9
    for tr in r.traces:
        assert tr.end in ("absorb", "leak", "alive", "cap", "truncated")
        assert all(abs(c) <= h for st in tr.steps for c in st[:3])
        assert all(st[5] in ("elastic", "react", "absorb", "leak", "alive", "cap") for st in tr.steps)
    ends = [t.end for t in r.traces]          # every source neutron was traced, so the end states add up to the totals
    assert ends.count("absorb") == r.absorbed and ends.count("leak") == r.leaked and ends.count("alive") + ends.count("cap") == r.alive_end
    assert sum(1 for t in r.traces if t.start_phase > 0) == r.n_born
