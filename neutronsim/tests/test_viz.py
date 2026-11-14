"""Visualizer logic tests. Pure functions only: no display or tkinter needed."""
import math
import random
from collections import Counter

import pytest

from neutronsim import channels, core, viz

ISOTOPES = ["B-10", "Al-27", "Ag-109", "U-235"]


def test_element_table_anchors():
    assert [viz.ELEMENTS.index(s) for s in ("H", "B", "Al", "Ag", "U")] == [1, 5, 13, 47, 92]


@pytest.mark.parametrize("iso", ISOTOPES)
@pytest.mark.parametrize("mt", sorted(viz.EMISSIONS))
def test_bookkeeping_conserves_charge_and_mass(iso, mt):
    _, z, a = viz.parse_isotope(iso)
    kind, emitted, res = viz.bookkeeping(z, a, mt)
    if kind == "emit":
        assert res[0] + sum(viz.SPECIES[s][0] for s in emitted) == z            # charge
        assert res[1] + sum(viz.SPECIES[s][1] for s in emitted) == a + 1        # nucleons (target + neutron)
        assert res[0] >= 1 and res[1] - res[0] >= 0
    else:
        assert kind == "other"                                                  # refused rather than invented


def test_known_textbook_cases():
    z, a = 5, 10
    assert viz.bookkeeping(z, a, 107)[2] == (3, 7)          # B-10(n,alpha)Li-7
    assert viz.bookkeeping(z, a, 103)[2] == (4, 10)         # B-10(n,p)Be-10
    assert viz.bookkeeping(13, 27, 102)[2] == (13, 28)      # Al-27(n,gamma)Al-28
    assert viz.bookkeeping(13, 27, 16)[2] == (13, 26)       # Al-27(n,2n)Al-26
    assert viz.bookkeeping(92, 235, 18) == ("fission", [], None)   # no products are invented for fission


def test_every_emission_mt_is_a_named_channel():
    assert set(viz.EMISSIONS) <= set(channels.NAMES)


@pytest.mark.parametrize("iso", ISOTOPES)
def test_nucleus_has_right_nucleon_counts(iso):
    _, z, a = viz.parse_isotope(iso)
    n = viz.make_nucleus(z, a, 0.0)
    assert n.types[:a].count("p") == z and n.types[:a].count("n") == a - z
    assert len(n.pos) == a + 1 and n.types[a] == "n"


@pytest.mark.parametrize("iso", ISOTOPES)
def test_every_channel_of_every_isotope_animates_cleanly(iso):
    ds = core.Dataset(iso)
    mts = [2] + ds.partial_mts + [None]
    for mt in mts:
        ev = viz.build_event(iso, mt, "x", random.Random(mt or 0))
        for i in range(61):
            ts = ev.duration * i / 60
            for p in viz.scene(ev, ts):
                assert p[0] in ("circle", "line", "rect", "text")
                nums = [p[1], p[2]] if p[0] in ("circle", "text") else (p[1] if p[0] == "line" else p[1:5])
                assert all(math.isfinite(v) for v in nums)


def test_emitted_particles_are_taken_from_the_nucleus_without_duplication():
    for mt in (16, 22, 24, 107, 117):
        ev = viz.build_event("Al-27", mt, "x", random.Random(1))
        if ev.kind != "emit":
            continue
        idx = [i for g in ev.groups for i, _ in g["members"]]
        assert len(idx) == len(set(idx))
        z_out = sum(1 for g in ev.groups for _, t in g["members"] if t == "p")
        assert z_out == 13 - ev.residual[0]
        assert len(ev.nuc.pos) - len(idx) == ev.residual[1]


def test_fission_is_drawn_without_inventing_numbers():
    ev = viz.build_event("U-235", 18, "x", random.Random(2))
    texts = " ".join(p[3] for p in viz.scene(ev, ev.duration) if p[0] == "text")
    assert "not modelled" in texts and "two fragments" in texts
    assert ev.groups == [] and ev.residual is None          # no fragment identities or neutron counts


def test_scene_always_carries_the_schematic_notice():
    ev = viz.build_event("B-10", 2, "x", random.Random(0))
    for ts in (0.0, 2.0, ev.duration):
        assert any(p[0] == "text" and "Schematic" in p[3] for p in viz.scene(ev, ts))


def test_outcome_table_matches_the_sampler():
    res = core.evaluate(core.Dataset("U-235"), 1e6)
    table = viz.outcome_table(res)
    assert math.isclose(sum(p for _, p in table), 1.0, rel_tol=1e-9)
    keys = {r["outcome"] for r in core.sample_outcomes(res, 1000, 1)["rows"]}
    assert {k for k, _ in table} == keys


def test_fire_draws_from_model_probabilities():
    res = core.evaluate(core.Dataset("B-10"), 0.0253)
    rng = random.Random(5)
    keys = Counter(viz.fire(res, rng).key for _ in range(400))
    table = dict(viz.outcome_table(res))
    assert set(keys) <= set(table)
    top = max(table, key=table.get)
    assert keys[top] / 400 == pytest.approx(table[top], abs=0.08)      # B-10 at thermal is (n,alpha)-dominated


def test_tally_scene_draws_bars_and_ticks():
    res = core.evaluate(core.Dataset("B-10"), 0.0253)
    table = viz.outcome_table(res)
    counts = Counter({table[0][0]: 3})
    prims = viz.tally_scene(table, counts, 3)
    assert sum(1 for p in prims if p[0] == "rect") == 2 * len(table)
    assert any(p[0] == "line" for p in prims)


def test_unsupported_isotope_name_is_rejected():
    with pytest.raises(core.InputError):
        viz.parse_isotope("Xx-12")


def test_cli_viz_registered_and_requires_energy(capsys):
    from neutronsim import cli
    assert cli.build_parser().parse_args(["viz", "B-10", "--energy", "1"]).cmd == "viz"
    with pytest.raises(SystemExit):
        cli.main(["viz", "B-10"])
