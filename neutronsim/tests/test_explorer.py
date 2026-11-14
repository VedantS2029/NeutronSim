"""Explorer model (pure logic), layout maths, and a headless smoke test of the window against the stub Tk."""
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from tk_stub import load_source, stubbed_tk
from neutronsim import explorer as ex
from neutronsim.nuclide import MAX_Z, Nuclide, periodic_position

SRC = Path(__file__).resolve().parents[1] / "src" / "neutronsim" / "explorer.py"


def test_typical_table_covers_every_element_sensibly():
    assert len(ex.TYPICAL_A) == MAX_Z
    for z in range(1, MAX_Z + 1):
        n = ex.typical_isotope(z)
        assert n.Z == z and n.charge == 0 and n.N >= 0
        if z > 2:
            assert 1.0 <= n.N / z <= 1.8, (z, n.key)          # sanity band: no wildly neutron-poor/rich defaults
    spot = {1: "H-1", 6: "C-12", 8: "O-16", 26: "Fe-56", 79: "Au-197", 82: "Pb-208", 92: "U-238"}
    for z, key in spot.items():
        assert ex.typical_isotope(z).key == key


def test_every_element_has_a_category_and_colour():
    cats = {ex.category(z) for z in range(1, MAX_Z + 1)}
    assert cats == set(ex.CATEGORY_COLOURS)
    assert ex.category(2) == ex.category(118) == "noble gas" and ex.category(26) == "transition metal"
    assert ex.category(57) == "lanthanide" and ex.category(92) == "actinide" and ex.category(11) == "alkali metal"


def test_layout_click_roundtrip_and_gaps():
    for z in range(1, MAX_Z + 1):
        x, y = ex.cell_origin(z)
        assert ex.cell_at(x + 5, y + 5) == z and ex.cell_at(x + ex.CELL - 1, y + ex.CELL - 1) == z
    assert ex.cell_at(0, 0) is None and ex.cell_at(ex.PAD + 5 * ex.CELL, ex.PAD + 7 * ex.CELL + 3) is None  # row-8 gap
    assert ex.cell_at(ex.PAD + 5 * ex.CELL, ex.PAD + 0 * ex.CELL + 3) is None                         # H/He row, empty middle
    assert max(ex.cell_origin(z)[0] for z in range(1, 119)) + ex.CELL <= ex.CANVAS_W


def test_select_and_steps_report_what_changed():
    m = ex.ExplorerModel(data_keys=frozenset())
    assert m.select_element(8) and m.nuclide.key == "O-16" and "element changed" in m.message
    assert m.step_neutrons(+1) and m.nuclide.key == "O-17" and "isotope changed: O-16 -> O-17" in m.message
    assert m.step_electrons(+2) and m.nuclide.charge == -2 and m.nuclide.key == "O-17" and "charge is now -2" in m.message
    assert m.step_protons(+1) and m.nuclide.symbol == "F" and m.nuclide.electrons == 10 and m.nuclide.charge == -1
    assert m.neutralise() and m.nuclide.charge == 0 and m.nuclide.electrons == 9


def test_impossible_steps_change_nothing_and_explain():
    m = ex.ExplorerModel(nuclide=Nuclide(1, 0), data_keys=frozenset())
    before = m.nuclide
    assert not m.step_protons(-1) and m.nuclide == before and "proton count" in m.message
    assert not m.step_neutrons(-1) and m.nuclide == before and "negative" in m.message
    assert not m.step_electrons(-2) and m.nuclide == before
    for _ in range(3):
        assert m.step_electrons(+1)
    assert not m.step_electrons(+1) and m.nuclide.electrons == 4
    assert not m.select_element(0) and not m.select_element(119) and m.nuclide.Z == 1
    assert not m.set_from_text("nonsense") and m.nuclide.Z == 1


def test_set_from_text_and_data_flag():
    m = ex.ExplorerModel()
    assert set(m.data_keys) == {"B-10", "Al-27", "Ag-109", "U-235"}
    assert m.set_from_text("U-235") and m.has_data and "Real evaluated" in "\n".join(m.status_lines())
    assert m.step_neutrons(+1) and not m.has_data and "No measured neutron data loaded for U-236" in "\n".join(m.status_lines())
    assert m.elements_with_data() == {5, 13, 47, 92}
    assert m.set_from_text("O-16 2-") and m.nuclide.charge == -2


def test_model_state_is_never_left_inconsistent():
    m = ex.ExplorerModel(data_keys=frozenset())
    for step in (m.step_protons, m.step_neutrons, m.step_electrons):
        for d in (-200, +200, -1, +1, 0):
            step(d)
            n = m.nuclide
            assert 1 <= n.Z <= MAX_Z and n.N >= 0 and 0 <= n.electrons <= n.Z + 3


# ---------------------------------------------------------------- window wiring against the stub
@pytest.fixture
def mod():
    with stubbed_tk():
        try:
            yield load_source("neutronsim.explorer_stubbed", SRC)
        finally:
            sys.modules.pop("neutronsim.explorer_stubbed", None)


def test_window_builds_draws_all_cells_and_responds(mod):
    app = mod.ExplorerApp(mod.ExplorerModel(data_keys=frozenset({"U-235"})))
    assert app.canvas.count("create_rectangle") == 118 and app.canvas.count("create_oval") == 1   # one marked element (U)
    x, y = mod.cell_origin(26)
    app._on_click(SimpleNamespace(x=x + 3, y=y + 3))
    assert app.model.nuclide.key == "Fe-56" and app.count_vars["protons"].get() == "26"
    app._on_click(SimpleNamespace(x=1, y=1))                                         # gap: nothing changes
    assert app.model.nuclide.key == "Fe-56"
    app._bump("neutrons", +1); app._bump("electrons", -3)
    assert app.model.nuclide.key == "Fe-57" and app.model.nuclide.charge == 3
    assert app.count_vars["neutrons"].get() == "31" and app.count_vars["electrons"].get() == "23"
    app._neutralise()
    assert app.model.nuclide.charge == 0
    app.entry_var.set("tritium"); app._apply_text()
    assert app.model.nuclide.key == "H-3" and app.entry_var.get() == ""
    app.entry_var.set("zzz"); app._apply_text()
    assert app.entry_var.get() == "zzz" and "cannot read" in app.msg_var.get()


def test_main_without_tk_fails_cleanly(mod, capsys):
    mod.tk = None
    assert mod.main() == 2 and "tkinter is not available" in capsys.readouterr().err


def test_cli_registers_explore():
    from neutronsim import cli
    a = cli.build_parser().parse_args(["explore", "U-235"])
    assert a.isotope == "U-235" and a.fn is cli.cmd_explore


def test_main_reports_bad_input_before_needing_a_display(mod, capsys):
    mod.tk = None
    assert mod.main("nonsense") == 2 and "cannot read" in capsys.readouterr().err


def test_status_includes_model_estimate_for_any_nuclide():
    m = ex.ExplorerModel(data_keys=frozenset())
    m.set_from_text("Pb-208")
    text = "\n".join(m.status_lines())
    assert "MODEL ESTIMATE" in text and "No measured neutron data loaded for Pb-208" in text
    m2 = ex.ExplorerModel()
    m2.set_from_text("U-235")
    t2 = "\n".join(m2.status_lines())
    assert "Real evaluated neutron data is available for U-235" in t2 and "MODEL ESTIMATE" in t2


def test_cli_predict_and_modelcheck(capsys):
    from neutronsim import cli
    assert cli.main(["predict", "U-235"]) == 0
    out = capsys.readouterr().out
    assert "MODEL ESTIMATE" in out and "Measured neutron data in this project: YES" in out
    assert cli.main(["predict", "tritium"]) == 0 and "Measured neutron data in this project: no" in capsys.readouterr().out
    assert cli.main(["predict", "nonsense"]) == 2 and "cannot read" in capsys.readouterr().err
    assert cli.main(["modelcheck"]) == 0 and "Summary:" in capsys.readouterr().out


def test_clicking_an_element_with_data_selects_an_isotope_with_data():
    m = ex.ExplorerModel()
    for z in (5, 13, 47, 92):                    # B, Al, Ag, U: each has a green dot
        assert m.select_element(z) and m.has_data, (z, m.nuclide.key)
    m.select_element(6)                          # C: no data, falls back to the typical isotope
    assert m.nuclide.key == "C-12" and not m.has_data
