"""Physics model: formula identities, behaviour on known cases, and a regression guard on how wrong it is allowed to be."""
import math
import pytest
from neutronsim import physics as ph
from neutronsim.nuclide import Nuclide


def test_binding_formula_regression_values():
    # pinned values (computed from the documented coefficients); catches accidental coefficient edits
    assert ph.binding_energy(26, 30) == pytest.approx(490.6, abs=0.1)
    assert ph.binding_energy(92, 146) == pytest.approx(1809.5, abs=0.1)
    assert ph.binding_energy(1, 0) == 0.0


def test_pairing_term_signs():
    base = lambda z, n: ph.binding_energy(z, n) - (ph.A_V * (z + n) - ph.A_S * (z + n) ** (2 / 3)
                                                   - ph.A_C * z * (z - 1) / (z + n) ** (1 / 3) - ph.A_A * (n - z) ** 2 / (z + n))
    assert base(8, 8) == pytest.approx(ph.A_P / math.sqrt(16))        # even-even bonus
    assert base(7, 7) == pytest.approx(-ph.A_P / math.sqrt(14))       # odd-odd penalty
    assert base(7, 8) == pytest.approx(0.0)                           # odd A: none


@pytest.mark.parametrize("z,n", [(-1, 3), (3, -1), (0, 0)])
def test_binding_rejects_nonsense(z, n):
    with pytest.raises(ValueError):
        ph.binding_energy(z, n)


def test_q_identities():
    for z, n in [(92, 146), (27, 33), (6, 8), (50, 70), (20, 20)]:
        q = ph.decay_q(z, n)
        # beta+ and EC differ by exactly 2 electron masses
        assert q["EC"] - q["beta+"] == pytest.approx(2 * ph.M_E)
        # alpha: only binding energies matter (nucleon masses cancel)
        assert q["alpha"] == pytest.approx(ph.binding_energy(z - 2, n - 2) + ph.B_HE4 - ph.binding_energy(z, n))
        # emission Q = -separation energy
        sn, sp = ph.separation_energies(z, n)
        assert q["n emission"] == pytest.approx(-sn) and q["p emission"] == pytest.approx(-sp)
        # beta- of (Z,N) is the reverse of EC of (Z+1,N-1)
        assert q["beta-"] == pytest.approx(-ph.decay_q(z + 1, n - 1)["EC"])


def test_channels_without_daughters_are_none():
    q = ph.decay_q(1, 0)
    assert all(v is None for v in q.values())
    assert ph.decay_q(2, 2)["alpha"] is None                 # He-4 cannot alpha-decay
    assert ph.decay_q(1, 2)["beta+"] is None                 # no Z=0 daughter
    assert ph.decay_q(118, 176)["beta-"] is None             # Z=119 is outside the table


def test_valley_of_stability_is_sensible():
    # real lowest-mass isobars (A=40 is Ar-40, not Ca-40). Model is exact for these except A=120, where it says
    # Te (Z=52) instead of Sn (Z=50): a known miss, because the model has no shell effects (Z=50 is magic).
    for a, z_real in [(16, 8), (40, 18), (56, 26), (120, 50), (208, 82)]:
        assert abs(ph.lowest_mass_isobar(a) - z_real) <= 2, a
    assert ph.lowest_mass_isobar(120) == 52
    # Odd A sits on ONE mass parabola, so its valley moves steadily to higher Z. Even A has two parabolas
    # (even-even and odd-odd, offset by the pairing term), so it may step back by 1 - that is real physics, not a bug.
    zs = [ph.lowest_mass_isobar(a) for a in range(11, 260, 2)]
    assert all(b >= a for a, b in zip(zs, zs[1:]))


def test_binding_per_nucleon_peaks_in_iron_region():
    best = max(range(20, 250), key=lambda a: ph.binding_energy(ph.lowest_mass_isobar(a), a - ph.lowest_mass_isobar(a)) / a)
    assert 56 <= best <= 70


def test_verdicts_on_known_cases():
    v = lambda t: ph.estimate_nuclide(Nuclide.parse(t)).verdict
    assert v("U-238") == "unstable"                          # alpha, well above the model's error
    assert v("Ni-62") == "stable"
    assert v("O-10") == "particle-unbound"                   # far beyond the proton drip line
    assert v("H-1") == "stable" and v("He-4") == "stable"
    assert v("C-14") == "borderline"                         # real Q is only 0.16 MeV: model must not claim certainty
    assert v("H-3") == "borderline"
    e = ph.estimate_nuclide(Nuclide.parse("O-30"))
    assert e.verdict == "unstable" and "marginal" in e.verdict_text    # marginal n-emission must not be called firm


def test_render_is_labelled_as_model_and_lists_limits():
    text = "\n".join(ph.render(ph.estimate_nuclide(Nuclide.parse("U-238"))))
    assert "MODEL ESTIMATE" in text and "not measured data" in text and "Not modelled: half-lives" in text
    assert "alpha" in text and "<- allowed" in text
    assert "A < 20" in "\n".join(ph.render(ph.estimate_nuclide(Nuclide.parse("C-12"))))


def test_electrons_do_not_change_the_estimate():
    a = ph.estimate_nuclide(Nuclide.parse("Fe-56"))
    b = ph.estimate_nuclide(Nuclide.parse("Fe-56 3+"))
    assert a.binding == b.binding and a.q == b.q


def test_known_limits_stay_documented_and_bounded():
    """Guards the claims made in physics.py's docstring. If you change coefficients, update the docstring too."""
    lines, st = ph.check()
    assert st["Q_max_abs_err"] < ph.Q_UNCERTAINTY                      # the 'marginal' band really covers the observed error
    assert st["B_max_abs_err"] < 10.0
    assert st["Q_sign_correct"] >= 13 and st["Q_total"] == 17
    assert st["stable_called_unstable"] == 0
    assert any("Summary" in l for l in lines)
