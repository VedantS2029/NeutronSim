"""Step 6: neutron forecast for any nuclide. The key guard is test_claimed_ranges_hold_against_real_data."""
import math

import pytest

from neutronsim import cli, explorer as ex, nphys as nf, physics as ph
from neutronsim.nuclide import MAX_Z, Nuclide


def nu(t):
    return Nuclide.parse(t)


def test_wavelength_matches_independent_calculation_and_scaling():
    # hbar/sqrt(2 m E) for a thermal neutron on a (nearly) infinitely heavy target: 2.862e4 fm
    assert nf.lambda_bar_fm(0.0253, 100000) == pytest.approx(2.862e4, rel=3e-3)
    assert nf.lambda_bar_fm(0.0253 * 4, 56) == pytest.approx(nf.lambda_bar_fm(0.0253, 56) / 2, rel=1e-9)    # ~ E^-1/2
    assert nf.lambda_bar_fm(14e6, 235) == pytest.approx(1.22, abs=0.03)
    with pytest.raises(ValueError):
        nf.lambda_bar_fm(0, 10)


def test_sizes_and_limits():
    assert nf.radius_fm(8) == pytest.approx(2 * nf.R0_FM)
    assert nf.potential_scattering_b(27) == pytest.approx(4 * math.pi * (1.35 * 3) ** 2 / 100)
    # black-disk limit: lambda-bar -> 0 gives pi R^2
    assert nf.geometric_nonelastic_b(100, 1e15) == pytest.approx(math.pi * nf.radius_fm(100) ** 2 / 100, rel=1e-3)
    assert nf.geometric_nonelastic_b(100, 1e6) > nf.geometric_nonelastic_b(100, 1e8)                       # larger at lower energy


def test_closed_shell_flag():
    assert nf.is_closed_shell(82, 126) and nf.is_closed_shell(8, 8) and nf.is_closed_shell(54, 81)      # Xe-135 -> N+1 = 82
    assert not nf.is_closed_shell(11, 12) and not nf.is_closed_shell(92, 143)


def test_statistical_capture_pieces():
    assert nf.gamma_over_spacing(1, 0) is None                       # compound nucleus H-2: far too light
    g = nf.gamma_over_spacing(26, 30)
    assert g and 1e-6 < g < 1 and nf.gamma_over_spacing(26, 30) == g
    t1, t2 = nf.capture_raw_b(26, 30, 0.0253), nf.capture_raw_b(26, 30, 0.0253 * 4)
    assert t1 / t2 == pytest.approx(2.0)                             # 1/v law
    # average formula tends to the unitarity-like limit 2 pi^2 lambda-bar^2 (Gamma_g/D) when Gamma_n >> Gamma_g
    assert nf.capture_raw_b(92, 146, 1e5, "average") < nf.capture_raw_b(92, 146, 1e3, "average")
    d = nf.level_spacing_eV(92, 146)
    assert d == pytest.approx(20.9, rel=0.1)                         # the observable a = A/10 was chosen to reproduce


def test_calibration_is_sane_and_traceable():
    c, f, n = nf.calibration()
    assert 5 < c < 100 and 5 < f < 60 and n >= 20


def test_regimes_and_refusals():
    low = nf.forecast(nu("Na-23"), 0.0253)
    assert "low energy" in low.regime and {r.name for r in low.rows} >= {"elastic (potential scattering)", "capture (n,gamma)"}
    mid = nf.forecast(nu("Fe-56"), 2e6)
    assert "LOW CONFIDENCE" in mid.regime and "Hauser-Feshbach" in mid.regime
    infl = next(r for r in mid.rows if r.name.startswith("inelastic"))
    assert infl.value_b and infl.value_b > 0 and infl.factor >= 1 and 4 in mid.channels
    fast = nf.forecast(nu("Na-23"), 14e6)
    tot = next(r for r in fast.rows if r.name == "total").value_b
    non = next(r for r in fast.rows if r.name.startswith("nonelastic")).value_b
    el = next(r for r in fast.rows if r.name.startswith("elastic")).value_b
    assert tot == pytest.approx(el + non, rel=1e-9) and "fast" in fast.regime
    geo = nf.forecast(nu("Na-23"), 30e6)                              # above 20 MeV: geometric fallback
    assert next(r for r in geo.rows if r.name == "total").value_b == pytest.approx(
        2 * next(r for r in geo.rows if r.name.startswith("nonelastic")).value_b) and not geo.channels
    light = next(r for r in nf.forecast(nu("H-1"), 1e6).rows if r.name.startswith("inelastic"))
    assert light.value_b is None and "too light" in light.note
    with pytest.raises(ValueError):
        nf.forecast(nu("Na-23"), -1.0)


def test_pb208_real_capture_probability_lies_inside_the_stated_range():
    from neutronsim import estsim
    fc = nf.forecast(nu("Pb-208"), 0.0253)
    lo, hi = estsim.prob_ranges(fc)[102]
    real = 4.9e-4 / (4.9e-4 + next(r.value_b for r in fc.rows if r.name.startswith("elastic")))
    assert lo < real < hi


def test_closed_shell_and_unpredictable_cases_are_labelled():
    pb = nf.forecast(nu("Pb-208"), 0.0253)
    assert pb.closed_shell and any("closed-shell" in t for t in pb.notes)
    cap = next(r for r in pb.rows if r.name.startswith("capture"))
    assert cap.factor > 100                                           # honest: orders of magnitude, not a tidy range
    h1 = next(r for r in nf.forecast(nu("H-1"), 0.0253).rows if r.name.startswith("capture"))
    assert h1.value_b is None and "too weakly bound" in h1.note


def test_after_capture_and_scope_lines():
    text = "\n".join(nf.render(nf.forecast(nu("Na-23"), 0.0253)))
    assert "after capture: Na-24" in text and "beta-" in text and "decay speed is not modelled" in text
    assert "MODEL ESTIMATE" in text and "Not modelled: fission" in text
    assert "fission" not in "\n".join(r.name for r in nf.forecast(nu("U-235"), 14e6).rows).lower()      # no fission numbers, by design


def test_ions_do_not_change_the_forecast():
    for e in (0.0253, 1e3, 14e6):
        assert nf.forecast(nu("Fe-56"), e) == nf.forecast(nu("Fe-56 3+"), e)


def test_every_element_every_regime_gives_finite_output():
    """'Any isotope of any element' means no crash and no nonsense numbers anywhere in the table."""
    for z in range(1, MAX_Z + 1):
        n = ex.typical_isotope(z)
        for e in (0.0253, 1e3, 1e6, 14e6):
            fc = nf.forecast(n, e)
            assert fc.rows
            for r in fc.rows:
                assert r.value_b is None or (math.isfinite(r.value_b) and r.value_b > 0), (n.key, e, r.name)
                assert r.factor is None or r.factor >= 1
            nf.render(fc)


def test_claimed_ranges_hold_against_real_data():
    """If this fails, either the physics changed or the error claims in nphys.py are now wrong: update the claims, not the test."""
    lines, st = nf.check()
    inside = lambda vals, f: all(1 / f <= v <= f for v in vals)
    assert inside(st["el_low"], nf.ELASTIC_FACTOR)
    for g, f in nf.HF_FACTORS.items():                                 # optical model + Hauser-Feshbach rows, heavier nuclides
        if g != "light":
            assert inside(st["hf"]["main"][g], f), g
            assert inside(st["hf"]["light"][g], nf.HF_FACTORS["light"]), g
    assert st["hf"]["n_cases"] > 30
    assert inside(st["nonel"], nf.FAST_FACTOR) and inside(st["tot_fast"], nf.FAST_FACTOR)
    assert inside(st["el_fast"], nf.FAST_ELASTIC_FACTOR)
    assert inside(st["cap_dense"], nf.CAPTURE_DENSE_FACTOR) and inside(st["cap_sparse"], nf.CAPTURE_SPARSE_FACTOR)
    assert st["spacing"]["U-238"] == pytest.approx(1.0, rel=0.1)
    assert st["spacing"]["Au-197"] < 0.3                              # known miss is reported, not hidden
    assert st["within"] / st["total"] >= 0.6
    assert st["worst_shell"] > 3                                       # closed shells really are off by orders of magnitude
    assert st["slope"] > 0                                             # documented heavy-nuclide overprediction
    assert any("in-sample" in l for l in lines) and any("not valid there" in l for l in lines)


def test_cli_forecast(capsys):
    assert cli.main(["forecast", "Na-23"]) == 0
    out = capsys.readouterr().out
    assert "FORECAST for a neutron of 0.0253 eV hitting Na-23" in out and "after capture" in out
    assert cli.main(["forecast", "U-235", "--energy", "14", "--unit", "MeV"]) == 0
    assert "Real measured data exists for U-235" in capsys.readouterr().out
    assert cli.main(["forecast", "nonsense"]) == 2
    assert cli.main(["forecastcheck"]) == 0 and "Summary of observed ratios" in capsys.readouterr().out


def test_explorer_shows_forecast_only_without_real_data():
    m = ex.ExplorerModel()
    m.set_from_text("Xe-136")
    assert "FORECAST for a neutron" in "\n".join(m.status_lines())
    m.set_from_text("U-235")
    text = "\n".join(m.status_lines())
    assert "FORECAST for a neutron" not in text and "Real evaluated neutron data is available" in text


def test_out_of_sample_14_MeV_values_inside_claimed_factors():
    _lines, oos = nf.hf_out_of_sample()
    for g, ratios in oos.items():
        assert ratios and all(1 / nf.HF_FACTORS[g] <= r <= nf.HF_FACTORS[g] for r in ratios), (g, ratios)
