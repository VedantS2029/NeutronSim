"""Step 5: chemistry module, explorer inside the wizard, and end-to-end CLI paths."""
import sys
import time
import threading
from pathlib import Path

import pytest

from tk_stub import load_source, stubbed_tk
from neutronsim import chem, cli
from neutronsim.nuclide import Nuclide

SRC = Path(__file__).resolve().parents[1] / "src" / "neutronsim"


def cfg(text):
    n = Nuclide.parse(text)
    return chem.config_string(chem.config(n.Z, n.electrons))


@pytest.mark.parametrize("text,spec", [
    ("H-1", "1s1"), ("He-4", "1s2"), ("C-12", "[He] 2s2 2p2"), ("O-16", "[He] 2s2 2p4"), ("Ne-20", "[Ne]"),
    ("Na-23", "[Ne] 3s1"), ("Fe-56", "[Ar] 3d6 4s2"), ("Ca-40", "[Ar] 4s2"), ("Ge-74", "[Ar] 3d10 4s2 4p2"),
    ("Cr-52", "[Ar] 3d5 4s1"), ("Cu-63", "[Ar] 3d10 4s1"), ("Pd-106", "[Kr] 4d10"), ("Ag-107", "[Kr] 4d10 5s1"),
    ("Au-197", "[Xe] 4f14 5d10 6s1"), ("U-238", "[Rn] 5f3 6d1 7s2"), ("Pb-208", "[Xe] 4f14 5d10 6s2 6p2"),
    ("Fe-56 3+", "[Ar] 3d5"), ("Fe-56 2+", "[Ar] 3d6"), ("Cu-63 +", "[Ar] 3d10"), ("Na-23 +", "[Ne]"),
    ("O-16 2-", "[Ne]"), ("Cl-35 -", "[Ar]"), ("C-12 -", "[He] 2s2 2p3"), ("U-238 91+", "1s1"),
])
def test_electron_configurations(text, spec):
    n = Nuclide.parse(text)
    assert chem.config(n.Z, n.electrons) == chem._expand(spec)


def test_config_string_formatting():
    assert cfg("Fe-56") == "[Ar] 3d6 4s2" and cfg("Fe-56 3+") == "[Ar] 3d5" and cfg("O-16 2-") == "[Ne]"
    assert cfg("He-4") == "[He]" and cfg("H-1") == "1s1" and cfg("H-1 +") == "(no electrons: bare nucleus)"
    assert cfg("Ca-40 2+") == "[Ar]"


def test_configs_always_conserve_electrons_and_fit_subshells():
    for z in range(1, 119):
        for e in sorted({0, 1, max(z - 3, 0), z - 1, z, z + 1, z + 3}):
            if e < 0:
                continue
            occ = chem.config(z, e)
            assert sum(occ.values()) == e, (z, e)
            assert all(0 < c <= chem.CAP[s] for s, c in occ.items()), (z, e)


def test_cations_lose_outer_shell_first():
    assert chem.config(26, 23) == {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 6, "3d": 5}     # Fe3+: 4s empties before 3d
    assert "4s" not in chem.config(26, 24)


def test_exception_table_is_self_consistent():
    for z, spec in chem._EXCEPTIONS.items():
        assert sum(chem._expand(spec).values()) == z, z


def test_masses_ions_and_hydrogenlike():
    lines, st = chem.check()
    assert st["max_abs"] < 0.01 and st["mean_abs"] < 0.006       # documented: model good to ~0.01 u
    na = Nuclide.parse("Na-23 +")
    assert chem.ion_mass_u(na) == pytest.approx(chem.mass_u(11, 12) - chem.M_E_U)
    assert chem.ion_mass_u(Nuclide.parse("Cl-35 -")) == pytest.approx(chem.mass_u(17, 18) + chem.M_E_U)
    bohr, dirac = chem.hydrogenlike_binding_eV(1)
    assert bohr == pytest.approx(13.6057, abs=1e-3) and dirac == pytest.approx(bohr, rel=1e-4)     # Dirac ~ Bohr for hydrogen
    b92, d92 = chem.hydrogenlike_binding_eV(92)
    assert b92 == pytest.approx(115_160, rel=1e-3) and d92 == pytest.approx(132_280, rel=1e-3)     # Dirac exceeds Bohr by ~15% at Z=92
    assert chem.hydrogenlike_binding_eV(2)[0] == pytest.approx(4 * 13.6057, rel=1e-4)               # scales as Z^2


def test_render_labels_what_it_does_not_know():
    t = "\n".join(chem.render(Nuclide.parse("Fe-56")))
    assert "not tabulated" in t and "[Ar] 3d6 4s2" in t
    assert "prediction, not measured" in "\n".join(chem.render(Nuclide.parse("Og-294")))
    assert "one-electron ion" in "\n".join(chem.render(Nuclide.parse("O-16 7+")))
    assert "exceptions" in "\n".join(chem.render(Nuclide.parse("Cu-63")))
    assert "exceptions" not in "\n".join(chem.render(Nuclide.parse("Cu-63 +")))


def test_cli_predict_and_modelcheck_include_chemistry(capsys):
    assert cli.main(["predict", "Fe-56 3+"]) == 0
    out = capsys.readouterr().out
    assert "[Ar] 3d5" in out and "ion mass" in out
    assert cli.main(["modelcheck"]) == 0 and "Isotope mass, model vs reference" in capsys.readouterr().out


# ---------------------------------------------------------------- explorer inside the wizard
@pytest.fixture
def both():
    with stubbed_tk():
        try:
            yield (load_source("neutronsim.gui_stubbed3", SRC / "gui.py"),
                   load_source("neutronsim.explorer_stubbed3", SRC / "explorer.py"))
        finally:
            for n in ("gui_stubbed3", "explorer_stubbed3"):
                sys.modules.pop(f"neutronsim.{n}", None)


def drain(app, seconds=15):
    end = time.time() + seconds
    while time.time() < end:
        app._poll()
        if app._q.empty() and not any(t.name.startswith("Thread") and t.is_alive() for t in threading.enumerate()):
            break
        time.sleep(0.05)
    app._poll()


def to_isotope_page(gui):
    app = gui.WizardApp()
    app.page.var.set(True)
    app.page._sync()
    app.next()
    assert isinstance(app.page, gui.IsotopePage)
    return app


def test_wizard_explorer_hands_back_an_isotope_with_data(both):
    gui, exmod = both
    gui._explorer = lambda: exmod
    app = to_isotope_page(gui)
    app.open_explorer()
    win = app.explorer
    assert isinstance(win, exmod.ExplorerWindow) and win.use_btn is not None
    win.model.set_from_text("Ag-109")
    win._refresh()
    assert not win.use_btn.disabled
    win._use()
    drain(app)
    assert app.model.isotope == "Ag-109" and not app.next_btn.disabled


def test_wizard_explorer_hands_back_isotope_without_data_as_model_estimate(both):
    gui, exmod = both
    gui._explorer = lambda: exmod
    app = to_isotope_page(gui)
    app.open_explorer()
    win = app.explorer
    win.model.set_from_text("C-14")
    win._refresh()
    assert not win.use_btn.disabled            # never a dead button
    win._use()
    drain(app)
    assert app.model.isotope == "C-14" and not app.model.has_data and not app.next_btn.disabled


def test_wizard_explorer_starts_on_current_selection(both):
    gui, exmod = both
    gui._explorer = lambda: exmod
    app = to_isotope_page(gui)
    app.page.choose("B-10")
    drain(app)
    app.open_explorer()
    assert app.explorer.model.nuclide.key == "B-10" and app.explorer.model.message == ""


def test_standalone_explorer_has_no_wizard_button(both):
    _, exmod = both
    assert exmod.ExplorerApp(exmod.ExplorerModel()).use_btn is None
