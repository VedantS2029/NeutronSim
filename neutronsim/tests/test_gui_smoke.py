"""Headless smoke test of the wizard's page flow against a stub Tk (see tk_stub.py)."""
import sys
import time
import threading
from pathlib import Path

import pytest

from tk_stub import load_source, stubbed_tk

SRC = Path(__file__).resolve().parents[1] / "src" / "neutronsim"
GUI = SRC / "gui.py"
VIZ = SRC / "viz.py"


@pytest.fixture
def gui():
    with stubbed_tk():
        try:
            yield load_source("neutronsim.gui_stubbed", GUI)
        finally:
            sys.modules.pop("neutronsim.gui_stubbed", None)


@pytest.fixture
def viz():
    with stubbed_tk():
        try:
            yield load_source("neutronsim.viz_stubbed", VIZ)
        finally:
            sys.modules.pop("neutronsim.viz_stubbed", None)


def drain(app, seconds=15):
    """Let background threads finish, then deliver their results on the 'UI thread'."""
    end = time.time() + seconds
    while time.time() < end:
        app._poll()
        if app._q.empty() and not any(t.name.startswith("Thread") and t.is_alive()
                                      for t in threading.enumerate()):
            break
        time.sleep(0.05)
    app._poll()


def test_full_walkthrough(gui):
    app = gui.WizardApp()
    assert isinstance(app.page, gui.WelcomePage)
    assert app.next_btn.disabled                       # cannot continue before acknowledging
    app.page.var.set(True)
    app.page._sync()
    assert not app.next_btn.disabled
    app.next()
    assert isinstance(app.page, gui.IsotopePage)
    assert app.next_btn.disabled                       # no isotope chosen yet
    app.page._select()                                 # stub listbox selects index 0
    drain(app)
    assert app.model.isotope == app.page.names[0]
    app.next()
    assert isinstance(app.page, gui.EnergyPage)
    app.page.e_var.set("-5")
    app.next()
    assert app.msg.get() and isinstance(app.page, gui.EnergyPage)   # bad energy blocks navigation
    app.page._preset("0.0253", "eV")
    app.next()
    assert isinstance(app.page, gui.ModePage)
    app.page.mode.set("sample")
    app.page.n.set("abc")
    app.next()
    assert isinstance(app.page, gui.ModePage)                        # bad sample count blocks
    app.page.n.set("2000")
    app.page.seed.set("7")
    app.next()
    assert isinstance(app.page, gui.ReviewPage)
    app.next()
    assert isinstance(app.page, gui.ResultsPage)
    drain(app)
    assert app.page.result is not None and app.page.result.supported
    assert "EDUCATIONAL MODEL" in app.page.out.text and "random draws" in app.page.out.text
    assert app.back_btn.disabled
    assert not app.page.vbtn.disabled                                # Visualize unlocks after a supported run
    opened = []
    fake = type("V", (), {"open_window": staticmethod(lambda parent, res: opened.append(res))})
    gui._viz = lambda: fake
    app.page.visualize()
    assert len(opened) == 1 and opened[0].isotope == app.model.isotope
    app.next()                                                       # "Close"
    assert getattr(app, "closed", False)


def test_poll_survives_callback_on_destroyed_widget(gui):
    app = gui.WizardApp()
    n_before = len(app.pending)

    def boom(_r):
        raise gui.tk.TclError("widget gone")
    app._q.put((boom, None))
    app._poll()
    assert len(app.pending) > n_before                  # loop re-armed itself


def test_start_over_returns_to_isotope_with_ack_kept(gui):
    app = gui.WizardApp()
    app.restart()
    assert isinstance(app.page, gui.IsotopePage) and app.model.acknowledged and not app.model.isotope


def test_fonts_are_font_objects_not_named_family_tuples(gui):
    src = GUI.read_text()
    assert 'font=("Tk' not in src      # tuple form with a named font is invalid on macOS


def test_viz_panel_fire_and_tally(viz):
    from neutronsim import core
    res = core.evaluate(core.Dataset("B-10"), 0.0253)
    panel = viz.VizPanel(viz.tk.Toplevel(None), res)
    assert panel.persist.get() is True                               # default: continue from the product
    panel.persist.set(False)                                         # old behaviour: always the same target nucleus
    assert panel.canvas.count("create_oval") > 0                     # idle nucleus was painted
    for _ in range(25):
        panel.fire()
        assert panel.event is not None
        panel._advance(panel.event.duration * 0.5)                   # mid-animation frame
        panel._advance(panel.event.duration + 0.01)                  # finish: tally updates, event clears
        assert panel.event is None
    assert panel.total == 25 and sum(panel.counts.values()) == 25
    assert set(panel.counts) <= {k for k, _ in panel.table}
    panel.reset()
    assert panel.total == 0


def _force(viz, panel, mt):
    """Fire one neutron whose outcome is forced to channel mt, and run its animation to the end."""
    import random
    panel.event = viz.build_event(panel.res.isotope, mt, f"MT{mt} forced", random.Random(1))
    panel._advance(panel.event.duration + 0.01)


def test_viz_panel_continues_from_the_product_by_default(viz):
    from neutronsim import core
    res = core.evaluate(core.Dataset("B-10"), 0.0253)
    panel = viz.VizPanel(viz.tk.Toplevel(None), res)
    assert panel.persist.get() is True
    _force(viz, panel, 2)                                            # elastic: same nucleus, tally kept
    assert panel.res.isotope == "B-10" and panel.total == 1
    _force(viz, panel, 102)                                          # capture: B-10 + n -> B-11, and it stays B-11
    assert panel.res.isotope == "B-11" and panel.idle.iso == "B-11" and panel.total == 0
    assert panel.chain == ["B-10", "B-11"] and "B-10 \u2192 B-11" in panel.msg.get()
    assert panel.table and set(k for k, _ in panel.table)           # a new outcome table for B-11 (model estimate)
    panel.fire()                                                     # the next shot is fired at B-11
    assert panel.event is not None and panel.event.iso == "B-11"
    panel._advance(panel.event.duration + 0.01)
    panel.restart()                                                  # back to the start nucleus; its tally is still there
    assert panel.res.isotope == "B-10" and panel.total == 2         # the elastic and the capture shot fired at B-10


def test_viz_next_isotope_rules(viz):
    import random
    ev = lambda iso, mt: viz.build_event(iso, mt, "k", random.Random(2))
    assert viz.next_isotope(ev("Fe-56", 102), "Fe-56", True)[0] == "Fe-57"
    assert viz.next_isotope(ev("Fe-57", 16), "Fe-56", True)[0] == "Fe-56"           # (n,2n) from Fe-57 -> Fe-56
    assert viz.next_isotope(ev("Fe-56", 103), "Fe-56", True)[0] == "Mn-56"          # (n,p)
    assert viz.next_isotope(ev("Fe-56", 4), "Fe-56", True)[0] == "Fe-56"            # scattering leaves it as it is
    assert viz.next_isotope(ev("Fe-57", 102), "Fe-56", False)[0] == "Fe-56"         # setting off: always the start nucleus
    iso, note = viz.next_isotope(ev("U-235", 18), "U-235", True)
    assert iso == "U-235" and "fission" in note
    assert viz.next_isotope(ev("Fe-57", None), "Fe-56", True)[0] == "Fe-57"         # unattributed: left alone


def test_viz_panel_survives_window_closing(viz):
    from neutronsim import core
    res = core.evaluate(core.Dataset("B-10"), 0.0253)
    panel = viz.VizPanel(viz.tk.Toplevel(None), res)
    panel.canvas.delete = lambda *_: (_ for _ in ()).throw(viz.tk.TclError("gone"))
    panel._advance(0.0)                                              # must not raise
    assert panel._alive is False


def test_viz_panel_with_no_outcomes_shows_nucleus_but_cannot_fire(viz):
    from neutronsim import estsim
    from neutronsim.nuclide import Nuclide
    res = estsim.visual_result(Nuclide.parse("H-1"), 1e6)
    panel = viz.VizPanel(viz.tk.Toplevel(None), res)
    assert panel.fire_btn.disabled and "too light" in panel.msg.get()
    panel.fire()
    assert panel.event is None and panel.total == 0


@pytest.mark.parametrize("mode,energy,unit,fireable", [
    ("xs", "0.0253", "eV", True),          # the case that used to leave Visualize locked
    ("sample", "0.0253", "eV", True),
    ("xs", "14", "MeV", True),
    ("xs", "1", "MeV", True),              # 1 keV - 20 MeV: Hauser-Feshbach channels
])
def test_visualize_unlocks_for_isotope_without_data(gui, mode, energy, unit, fireable):
    app = gui.WizardApp()
    app.page.var.set(True); app.page._sync(); app.next()
    app.page.choose("Na-23")                           # no evaluated data for Na-23
    drain(app)
    assert app.model.isotope == "Na-23" and not app.model.has_data
    app.next()
    app.page._preset(energy, unit)
    app.next()
    app.page.mode.set(mode); app.page._changed()
    app.next(); app.next()
    assert isinstance(app.page, gui.ResultsPage)
    drain(app)
    assert "MODEL ESTIMATE" in app.page.out.text and not app.page.vbtn.disabled
    opened = []
    gui._viz = lambda: type("V", (), {"open_window": staticmethod(lambda parent, res: opened.append(res))})
    app.page.visualize()
    assert len(opened) == 1 and opened[0].isotope == "Na-23" and bool(opened[0].channels) is fireable
