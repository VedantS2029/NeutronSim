"""Box visualizer (pure scene logic; the tkinter panel is covered through test_boxgui)."""
import math

import pytest

from neutronsim import box, boxviz


@pytest.fixture(scope="module")
def res():
    return box.run_box("Fe-56", 14e6, density_factor=0.3, n=80, seed=4, trace=80)


def test_projection_keeps_the_cube_on_the_canvas_from_any_angle():
    h = 5.0
    for yaw in (0.0, 0.6, 1.9, 3.5):
        for pitch in (-1.2, 0.0, 0.45, 1.2):
            for c in boxviz.cube_corners(h):
                sx, sy, _ = boxviz.project(*c, yaw, pitch, h)
                assert 0 <= sx <= boxviz.W and 0 <= sy <= boxviz.H
    assert len(boxviz.CUBE_EDGES) == 12


def test_energy_colour_is_a_hex_colour_and_thermal_differs_from_fast():
    for e in (0.0, 1e-5, 0.0253, 1e3, 1e6, 2e7):
        c = boxviz.energy_color(e)
        assert len(c) == 7 and c[0] == "#" and int(c[1:], 16) >= 0
    assert boxviz.energy_color(0.0253) != boxviz.energy_color(14e6)


def test_struck_sites_are_real_collision_sites_and_only_finished_ones(res):
    end = boxviz.total_phase(res.traces) + 1
    sites = boxviz.struck_sites(res.traces, end)
    assert len(sites) == res.collisions                        # one ring per real collision, none invented
    h = res.side_cm / 2.0
    assert all(abs(c) <= h + 1e-9 for s_ in sites for c in s_[:3])
    assert {s_[3] for s_ in sites} <= {"elastic", "react", "absorb"}
    assert len(boxviz.struck_sites(res.traces, 0.5)) <= len(boxviz.struck_sites(res.traces, 3.0)) <= len(sites)
    assert not hasattr(boxviz, "atom_dots")                    # the old invented dots are gone


def test_scale_facts_are_consistent_with_the_run(res):
    f = boxviz.scale_facts(res)
    assert math.isclose(f["spacing_cm"] ** 3 * f["atoms_per_cm3"], 1.0, rel_tol=1e-9)
    assert math.isclose(f["atoms_in_box"], res.atoms_per_cm3 * res.side_cm ** 3, rel_tol=1e-12)
    assert math.isclose(f["mfp_in_spacings"] * f["spacing_cm"], res.mean_free_path_cm, rel_tol=1e-12)
    assert f["spacing_in_nuclear_diameters"] > 1e3             # atoms are mostly empty space: spacing >> nucleus


def test_path_length_matches_the_recorded_geometry(res):
    end = boxviz.total_phase(res.traces) + 1
    manual = 0.0
    for tr in res.traces:
        p = (tr.x0, tr.y0, tr.z0)
        for st in tr.steps:
            manual += math.dist(p, st[:3]); p = st[:3]
    assert math.isclose(boxviz.path_length_cm(res.traces, end), manual, rel_tol=1e-12)
    assert boxviz.path_length_cm(res.traces, 0.0) == 0.0


def test_wrap_never_exceeds_the_width():
    assert all(len(ln) <= 20 for ln in boxviz._wrap("one two three four five six seven eight nine ten", 20))


def test_scene_draws_rings_only_when_asked(res):
    on = [p for p in boxviz.scene(res, 8.0, 0.6, 0.45, True) if p[0] == "circle" and p[4] == "" and p[6] == 1 and p[5] in boxviz.KIND_COLOR.values() and p[3] == 2.6]
    off = [p for p in boxviz.scene(res, 8.0, 0.6, 0.45, False) if p[0] == "circle" and p[4] == "" and p[6] == 1 and p[5] in boxviz.KIND_COLOR.values() and p[3] == 2.6]
    assert len(on) > len(off)


def test_states_before_during_and_after(res):
    tr = next(t for t in res.traces if len(t.steps) >= 2 and t.start_phase == 0)
    mid = boxviz.neutron_state(tr, 0.5)
    assert mid["done"] == "" and mid["pos"] != (tr.x0, tr.y0, tr.z0)
    end = boxviz.neutron_state(tr, boxviz.total_phase(res.traces) + 5)
    assert end["done"] == tr.end and end["pos"] == tr.steps[-1][:3]
    child = next((t for t in res.traces if t.start_phase > 0), None)
    if child is not None:
        assert boxviz.neutron_state(child, child.start_phase - 0.5) is None


def test_final_tally_matches_the_run_because_every_source_neutron_is_traced(res):
    t = boxviz.tally_at(res.traces, boxviz.total_phase(res.traces) + 1)
    assert t["ends"]["absorb"] == res.absorbed and t["ends"]["leak"] == res.leaked
    assert t["ends"]["alive"] + t["ends"]["cap"] == res.alive_end
    assert sum(t["mts"].values()) == res.collisions
    early = boxviz.tally_at(res.traces, 0.5)
    assert sum(early["mts"].values()) <= sum(t["mts"].values())


def test_scene_and_tally_scene_primitives_are_well_formed(res):
    shapes = {"circle": 7, "line": 5, "rect": 7, "text": 8}
    for ph in (0.0, 1.3, boxviz.total_phase(res.traces) + 1):
        for prims in (boxviz.scene(res, ph, 0.6, 0.45), boxviz.tally_scene(res, ph)):
            assert prims
            for p in prims:
                assert p[0] in shapes and len(p) == shapes[p[0]], p
                if p[0] == "line":
                    assert len(p[1]) % 2 == 0 and all(math.isfinite(v) for v in p[1])
    texts = " ".join(p[3] for p in boxviz.scene(res, 0.0, 0.6, 0.45) if p[0] == "text")
    assert "Fe-56" in texts and "MODEL ESTIMATE" in texts and "schematic" in texts


def test_caption_names_the_latest_event(res):
    assert "start" in boxviz.caption(res.traces, 0.0)
    assert "Latest event" in boxviz.caption(res.traces, boxviz.total_phase(res.traces) + 1)


def test_data_isotope_scene_has_no_model_label():
    r = box.run_box("B-10", 0.0253, density_factor=0.01, n=30, seed=1, trace=30)
    texts = " ".join(p[3] for p in boxviz.scene(r, 3.0, 0.6, 0.45) if p[0] == "text")
    assert "MODEL ESTIMATE" not in texts


def test_2d_view_is_flat_square_and_ignores_rotation(res):
    a = boxviz.scene(res, 8.0, 0.6, 0.45, True, True, True)
    b = boxviz.scene(res, 8.0, 2.0, -1.0, True, True, True)
    assert a == b                                              # rotation has no effect in the flat view
    sq = [p for p in a if p[0] == "line" and p[2] == boxviz.EDGE_FRONT and len(p[1]) == 10]
    assert len(sq) == 1                                        # one square outline, no 12-edge cube
    xs, ys = sq[0][1][0::2], sq[0][1][1::2]
    assert all(0 <= x <= boxviz.W for x in xs) and all(0 <= y <= boxviz.H for y in ys)
