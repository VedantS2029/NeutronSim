"""Box window (step B2): model logic without a display, and the window wiring against the stub Tk."""
import sys
import time
import threading
from pathlib import Path

import pytest

from neutronsim import box, core
from neutronsim.boxgui import BoxModel, SLIDER_MAX, SLIDER_MIN, plot_series, stat_rows
from tk_stub import load_source, stubbed_tk

SRC = Path(__file__).resolve().parents[1] / "src" / "neutronsim"


def test_model_defaults_and_slider_range():
    m = BoxModel()
    assert m.isotope == "B-10" and m.isotopes[:3] == ["Ag-109", "Al-27", "B-10"] and "Fe-56" in m.isotopes
    m.set_slider(-99); assert m.slider == SLIDER_MIN
    m.set_slider(+99); assert m.slider == SLIDER_MAX
    assert m.density_factor == pytest.approx(box.MAX_DENSITY_FACTOR)
    assert "x solid" in m.density_text() and "atoms/cm3" in m.density_text()


def test_fissile_start_is_refused_with_a_reason():
    m = BoxModel("U-235")
    assert m.isotope in box.boxable_isotopes() and "fission" in m.initial_note
    assert "actinide" in BoxModel("Pu-239").initial_note


def test_any_other_isotope_is_accepted_and_typed_text_is_normalised():
    m = BoxModel("Fe-56")
    assert m.isotope == "Fe-56" and m.initial_note == ""
    m.set_isotope("pb208"); assert m.isotope == "Pb-208" and m.check() is None
    assert "x solid" in m.density_text()
    m.set_isotope("not an isotope"); assert m.check()
    m.set_isotope("U-238"); assert "actinide" in m.check()


def test_model_isotope_run_records_traces():
    m = BoxModel("Fe-56"); m.set_preset("14", "MeV"); m.n_text = "100"; m.trace_text = "25"; m.set_slider(-1.0)
    r = m.run()
    assert r.model_estimate and len(r.traces) >= 25            # 25 source neutrons + any (n,2n) children
    assert sum(1 for t in r.traces if t.start_phase == 0) >= 25
    m.trace_text = "0"; assert m.run().traces == []


@pytest.mark.parametrize("field,value,word", [("energy_text", "abc", "energy"), ("energy_text", "-1", "energy"),
                                              ("side_text", "0", "side"), ("side_text", "x", "side"),
                                              ("n_text", "0", "neutrons"), ("n_text", "1.5", "neutrons")])
def test_model_rejects_bad_input(field, value, word):
    m = BoxModel()
    setattr(m, field, value)
    assert word in m.check()
    with pytest.raises(core.UnsupportedError):
        m.run()


def test_run_stats_and_plot_series():
    m = BoxModel("Al-27")
    m.set_preset("14", "MeV"); m.n_text = "200"; m.set_slider(0.0)
    r = m.run()
    names = dict(stat_rows(r))
    assert "Left in the box (alive)" in names and "Heat deposited" in names and len(names) == 10
    s = plot_series(r)
    assert len(s["t"]) == len(s["alive"]) == len(s["absorbed"]) == len(s["leaked"]) == len(s["heat"]) >= 2
    assert s["alive"][-1] == pytest.approx(100.0 * r.alive_end / r.n_total)
    assert s["heat"][-1] == pytest.approx(r.heat_MeV)


def test_density_slider_changes_the_physics():
    m = BoxModel("B-10"); m.n_text = "400"
    m.set_slider(-4.0); thin = m.run()
    m.set_slider(0.0); thick = m.run()
    assert thin.leaked > thick.leaked and thick.heat_MeV > thin.heat_MeV


@pytest.fixture
def ui():
    import neutronsim
    with stubbed_tk():
        old = getattr(neutronsim, "boxviz", None)
        sys.modules.pop("neutronsim.boxviz", None)
        neutronsim.boxviz = load_source("neutronsim.boxviz", SRC / "boxviz.py")        # boxgui's lazy import finds this stubbed copy
        try:
            yield load_source("neutronsim.boxgui_stubbed", SRC / "boxgui.py")
        finally:
            sys.modules.pop("neutronsim.boxgui_stubbed", None)
            sys.modules.pop("neutronsim.boxviz", None)
            if old is not None:
                neutronsim.boxviz = old
                sys.modules["neutronsim.boxviz"] = old
            elif hasattr(neutronsim, "boxviz"):
                del neutronsim.boxviz


def _drain(win, seconds=30):
    end = time.time() + seconds
    while time.time() < end:
        win._poll()
        if win.model.result is not None and not win._busy:
            return
        time.sleep(0.05)
    raise AssertionError("box run did not finish")


def test_window_runs_and_fills_stats(ui):
    win = ui.BoxApp(ui.BoxModel("B-10"))
    win._run()
    _drain(win)
    assert win.vars["Absorbed"].get() != "-" and win.model.result.n_source == 500
    assert win.cv_pop.count("create_line") > 0 and win.cv_heat.count("create_line") > 0


def test_slider_reruns_and_only_the_latest_run_is_shown(ui):
    win = ui.BoxApp(ui.BoxModel("B-10"))
    win._slider_moved(-3.0)
    win._slider_moved(-1.0)
    assert win.model.slider == -1.0
    win._run()
    _drain(win)
    assert win.model.result.density_factor == pytest.approx(0.1)


def test_window_reports_bad_input_without_running(ui):
    win = ui.BoxApp(ui.BoxModel("B-10"))
    win.n_var.set("0")
    win._run()
    assert "neutrons" in win.msg.calls[-1][2]["text"]


def test_open_from_wizard_passes_isotope_and_energy():
    with stubbed_tk():
        gui = load_source("neutronsim.gui_stubbed2", SRC / "gui.py")
        try:
            import neutronsim
            old = getattr(neutronsim, "boxgui", None)
            old_viz = getattr(neutronsim, "boxviz", None)
            sys.modules.pop("neutronsim.boxviz", None)
            neutronsim.boxviz = load_source("neutronsim.boxviz", SRC / "boxviz.py")
            sys.modules.pop("neutronsim.boxgui", None)
            neutronsim.boxgui = load_source("neutronsim.boxgui", SRC / "boxgui.py")   # the wizard's lazy import finds this one
            app = gui.WizardApp()
            app.model.isotope = "Al-27"
            app.model.energy_text, app.model.unit = "1", "MeV"
            app.open_box()
            assert app.box_window.model.isotope == "Al-27"
            assert app.box_window.model.energy_text == "1e+06"
        finally:
            sys.modules.pop("neutronsim.gui_stubbed2", None)
            sys.modules.pop("neutronsim.boxgui", None)
            sys.modules.pop("neutronsim.boxviz", None)
            if old_viz is not None:
                neutronsim.boxviz = old_viz
                sys.modules["neutronsim.boxviz"] = old_viz
            elif hasattr(neutronsim, "boxviz"):
                del neutronsim.boxviz
            if old is not None:
                neutronsim.boxgui = old
            elif hasattr(neutronsim, "boxgui"):
                del neutronsim.boxgui


def test_window_has_watch_tab_fed_with_the_run(ui):
    win = ui.BoxApp(ui.BoxModel("Fe-56"))
    win.e_var.set("1"); win.u_var.set("MeV"); win.s_var.set("20"); win.n_var.set("80"); win.t_var.set("10")
    win._run()
    _drain(win)
    assert win.model.result.model_estimate and len(win.viz_panel.res.traces) == 10
    assert win.tabs.count("add") == 2 and "MODEL ESTIMATE" in win.model_note.calls[-1][2]["text"]


def test_typing_an_isotope_runs_it(ui):
    win = ui.BoxApp(ui.BoxModel("B-10"))
    win.iso_var.set("cu63"); win.e_var.set("1"); win.u_var.set("MeV")
    win._run()
    _drain(win)
    assert win.model.result.isotope == "Cu-63"


def test_window_can_open_on_the_watch_tab(ui):
    win = ui.BoxApp(ui.BoxModel("Al-27"), watch=True)
    assert win.tabs.count("select") == 1
