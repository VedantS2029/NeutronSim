"""Independent accuracy checks of the box engine (analytic results, not re-statements of the code)."""
import math
import random

import pytest

from neutronsim import box


def test_speed_is_relativistically_exact_and_hits_the_2200_m_s_thermal_value():
    assert box.speed_cm_s(0.0253) == pytest.approx(2.2e5, rel=1e-4)               # the classic 2200 m/s
    e = 2e7
    g = 1.0 + e / box.MN_C2_EV
    assert box.speed_cm_s(e) == pytest.approx(box.C_CM_S * math.sqrt(1.0 - 1.0 / g ** 2), rel=1e-12)
    assert box.speed_cm_s(e) < box.C_CM_S * math.sqrt(2 * e / box.MN_C2_EV)      # non-relativistic formula overshoots
    assert box.speed_cm_s(1e-5) > 0


@pytest.mark.parametrize("w", [1.0, -1.0, 1 - 1e-9, 0.999999, 0.3, -0.7])
def test_rotate_gives_the_requested_lab_cosine_even_near_the_poles(w):
    rng = random.Random(11)
    u, v = math.sqrt(max(0.0, 1 - w * w)), 0.0
    for mu in (-0.9, 0.0, 0.3, 0.99):
        a, b, c = box._rotate(u, v, w, mu, rng)
        assert a * u + b * v + c * w == pytest.approx(mu, abs=1e-9)
        assert a * a + b * b + c * c == pytest.approx(1.0, abs=1e-12)


def test_elastic_mean_energy_ratio_and_mean_lab_cosine_match_textbook_values():
    rng, A, N = random.Random(5), 26.7, 200_000
    se = sm = 0.0
    for _ in range(N):
        muc = 2 * rng.random() - 1
        f = A * A + 1 + 2 * A * muc
        se += f / (A + 1) ** 2
        sm += (1 + A * muc) / math.sqrt(f)
    alpha = ((A - 1) / (A + 1)) ** 2
    assert se / N == pytest.approx((1 + alpha) / 2, rel=2e-3)
    assert sm / N == pytest.approx(2.0 / (3.0 * A), abs=3e-3)


def test_uncollided_leakage_from_a_cube_centre_matches_the_analytic_average():
    n = 6000
    r = box.run_box("B-10", 1.0, side_cm=1.0, n=n, seed=3, trace=n, t_end_s=1.0, density_factor=0.02)
    mfp, h = r.mean_free_path_cm, 0.5
    unc = sum(1 for t in r.traces[:n] if len(t.steps) == 1 and t.steps[0][5] == "leak")
    rng, tot, m = random.Random(5), 0.0, 200_000
    for _ in range(m):
        u, v, w = box._isotropic(rng)
        tot += math.exp(-min(h / abs(q) for q in (u, v, w) if q != 0) / mfp)
    p = tot / m
    assert abs(unc / n - p) < 4.5 * math.sqrt(p * (1 - p) / n)


def test_every_neutron_is_accounted_for_and_energy_closes_at_low_and_high_energy():
    for iso, e in (("Al-27", 1e6), ("B-10", 0.0253), ("Ag-109", 1.4e7)):
        r = box.run_box(iso, e, side_cm=5.0, n=300, seed=8, density_factor=0.5)
        assert r.absorbed + r.leaked + r.alive_end == r.n_total
        assert abs(r.closure_MeV) < 1e-6 * max(1.0, r.energy_in_MeV)
