"""Optical model + Hauser-Feshbach (hf.py): physical limits, sum rules, thresholds, coverage."""
import math

import pytest

from neutronsim import explorer as ex, hf, nphys
from neutronsim.nuclide import MAX_Z, Nuclide


def test_riccati_functions_satisfy_the_wronskian():
    for l in (0, 1, 3, 6):
        for x in (0.05, 0.5, 2.0, 8.0):
            f, fp, g, gp = hf._bessel(l, x)
            assert f * gp - fp * g == pytest.approx(-1.0, abs=1e-6)


@pytest.mark.parametrize("z,a", [(13, 27), (47, 109), (92, 235), (26, 56)])
@pytest.mark.parametrize("e", [1e3, 1e5, 1e6, 5e6, 14e6])
def test_transmission_coefficients_between_0_and_1(z, a, e):
    ts = hf.transmission(a, z, e)
    assert ts and all(0.0 <= t <= 1.0 for t in ts)
    assert ts[0] >= ts[-1]


def test_black_disk_limit_at_high_energy():
    z, a = 82, 208
    sig = hf.optical_cross_sections(a, z, 100e6)["reaction"]
    disk = math.pi * (nphys.radius_fm(a) + nphys.lambda_bar_fm(100e6, a)) ** 2 / 100
    assert 0.6 < sig / disk < 1.4
    t = hf.transmission(a, z, 100e6)
    assert t[0] > 0.5                                           # low partial waves are absorbed almost completely


def test_optical_unitarity_and_total_cross_section():
    oc = hf.optical_cross_sections(56, 26, 2e6)
    assert oc["total"] == pytest.approx(oc["reaction"] + oc["shape_elastic"], rel=1e-6)


def test_charged_particle_barrier():
    assert hf.barrier_transmission(1, 1, 12, 27, 0.5, 0) < 1e-3          # far below the Coulomb barrier
    assert hf.barrier_transmission(1, 1, 12, 27, 15.0, 0) > 0.9          # far above it
    assert hf.barrier_transmission(2, 4, 12, 27, 5.0, 0) < hf.barrier_transmission(1, 1, 12, 27, 5.0, 0)
    assert hf.barrier_transmission(1, 1, 12, 27, 5.0, 4) < hf.barrier_transmission(1, 1, 12, 27, 5.0, 0)   # centrifugal term raises it


@pytest.mark.parametrize("t,e", [("Fe-56", 2e6), ("Fe-56", 14e6), ("Ag-109", 1e5), ("Pb-208", 14e6), ("Na-23", 1e4)])
def test_sum_rule_channels_add_up_to_compound_nucleus(t, e):
    nu = Nuclide.parse(t)
    r = hf.decay(nu.Z, nu.N, e)
    assert sum(r.branches.values()) + r.compound_elastic == pytest.approx(1.0, abs=1e-5)
    assert sum(hf.channel_sigmas(r).values()) + r.sigma_cn_b * r.compound_elastic == pytest.approx(r.sigma_cn_b, rel=1e-5)
    assert r.elastic_b + sum(hf.channel_sigmas(r).values()) == pytest.approx(r.sigma_total_b, rel=1e-4)


def test_thresholds_channels_closed_below_their_q_value():
    nu = Nuclide.parse("Fe-56")
    th = hf.thresholds_mev(nu.Z, nu.N)
    for name, mt in (("(n,2n)", 16), ("(n,p)", 103), ("(n,alpha)", 107)):
        below = hf.channel_sigmas(hf.decay(nu.Z, nu.N, max(th[name] - 0.5, 0.01) * 1e6))
        assert below.get(mt, 0.0) < 1e-9, name
    above = hf.channel_sigmas(hf.decay(nu.Z, nu.N, 14e6))
    assert above[16] > 0 and above[103] > 0
    low = hf.channel_sigmas(hf.decay(nu.Z, nu.N, 1e5))
    assert low.get(4, 0.0) == 0.0                                  # below the first excited level only elastic and capture are open


def test_levels_and_fermi_gas_are_sensible():
    for z, n in ((26, 30), (13, 14), (92, 143), (11, 11), (82, 126)):
        lv = hf.discrete_levels(z, n)
        assert lv[0] == 0 and list(lv) == sorted(lv) and hf.continuum_start(z, n) > lv[-1]
    assert hf.first_excited_mev(82, 126) > hf.first_excited_mev(64, 92)      # doubly magic stiffer than a rotor


def test_every_element_gives_finite_channels_from_1_keV_to_20_MeV():
    for z in range(1, MAX_Z + 1):
        nu = ex.typical_isotope(z)
        if hf.available(nu.Z, nu.N):
            continue
        for e in (1.001e3, 1e5, 1e6, 4.9e6, 20e6):
            r = hf.decay(nu.Z, nu.N, e)
            vals = [r.sigma_cn_b, r.sigma_shape_b, r.sigma_total_b, *hf.channel_sigmas(r).values()]
            assert all(math.isfinite(v) and v >= 0 for v in vals), (nu.key, e)
    with pytest.raises(ValueError):
        hf.decay(26, 30, 30e6)


def test_estimated_first_levels_against_known_nuclides():
    lines, rms = hf.level_validation()
    assert rms["even-even"] < 0.35 and rms["odd-A"] < 1.0       # measured 0.26 and 0.84 dex; odd-A scatter is real, not a bug


def test_model_thresholds_against_known_reactions():
    _lines, rms = hf.threshold_validation()
    assert rms < 1.5                                            # measured ~1.0 MeV: liquid-drop masses have no shell effects
