"""Model vs real data: bookkeeping identities, agreement bounds against the shipped ENDF Q-values, honest 'no data' paths."""
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from tk_stub import load_source, stubbed_tk
from neutronsim import cli, compare as cp, core, physics as ph
from neutronsim import explorer as ex
from neutronsim.nuclide import Nuclide

ISOS = ["B-10", "Al-27", "Ag-109", "U-235"]
LIGHT_B = {"n": 0.0, "p": 0.0, "d": 2.224566, "t": 8.481798, "he3": 7.718058, "a": ph.B_HE4}


@pytest.mark.parametrize("z,n", [(13, 14), (47, 62), (92, 143), (5, 5), (82, 126)])
def test_reaction_q_equals_binding_energy_balance(z, n):
    """Independent bookkeeping: nucleon masses cancel, so Q = sum(B of products) - B(target) for every modelled MT."""
    checked = 0
    for mt, em in cp.EMISSIONS.items():
        q = cp.reaction_q(z, n, mt)
        if q is None:
            continue
        zd = z - sum(cp._ZN[p][0] for p in em)
        nd = n + 1 - sum(cp._ZN[p][1] for p in em)
        b_d = 0.0 if (zd, nd) == (0, 0) else ph.binding_energy(zd, nd)
        expect = b_d + sum(LIGHT_B[p] for p in em) - ph.binding_energy(z, n)
        assert q == pytest.approx(expect, abs=1e-6), (mt, z, n)
        checked += 1
    assert checked >= 8


def test_capture_and_n2n_identities():
    z, n = 92, 143
    assert cp.reaction_q(z, n, 102) == pytest.approx(ph.binding_energy(z, n + 1) - ph.binding_energy(z, n))
    sn, _ = ph.separation_energies(z, n)
    assert cp.reaction_q(z, n, 16) == pytest.approx(-sn)
    assert cp.reaction_q(z, n, 17) == pytest.approx(-(sn + ph.separation_energies(z, n - 1)[0]))


def test_unmodelled_or_impossible_channels_are_none():
    assert cp.reaction_q(92, 143, 18) is None and cp.reaction_q(92, 143, 4) is None and cp.reaction_q(92, 143, 999) is None
    assert cp.reaction_q(1, 0, 103) is None                  # (n,p) on hydrogen: no valid daughter
    assert cp.reaction_q(2, 2, 16) is not None               # (n,2n) on He-4 leaves He-3: valid


@pytest.mark.parametrize("iso", ISOS)
def test_model_tracks_evaluated_q_values(iso):
    rows = cp.q_rows(iso)
    assert rows and all(r.data_MeV != 0 for r in rows)
    assert max(abs(r.diff) for r in rows) < 3.0              # observed worst ~2.6 MeV (Al-27 (n,d+alpha)); documented in README
    assert sum(abs(r.diff) for r in rows) / len(rows) < 1.5


def test_known_rows_and_skips():
    rows = {r.mt: r for r in cp.q_rows("B-10")}
    assert set(rows) == {102, 104, 113}                      # (n,alpha) has Q stored as 0 = not given: must be skipped
    assert rows[102].data_MeV == pytest.approx(11.456, abs=1e-3)
    assert 107 not in {r.mt for r in cp.q_rows("B-10")}
    assert 113 in rows                                       # all-particles-emitted channel (no daughter) works
    assert 18 not in {r.mt for r in cp.q_rows("U-235")}      # fission never compared


def test_full_report_shows_model_only_rows_and_limits():
    text = "\n".join(cp.data_lines("B-10", 0.0253, full=True))
    assert "MODEL ONLY" in text and "(n,alpha)" in text and "Fission (MT18)" in text and "AGREES" in text


def test_lines_for_each_kind_of_nuclide():
    u = "\n".join(cp.lines(Nuclide.parse("U-235")))
    assert "DATA (ENDF/B-VIII.0" in u and "REFERENCE" in u and "DISCREPANT" in u          # discrepancy is shown, not hidden
    c14 = "\n".join(cp.lines(Nuclide.parse("C-14")))
    assert "DATA" not in c14 and "NOT allowed, but this decay is real" in c14             # wrong model sign is reported
    none = cp.lines(Nuclide.parse("Xe-136"))
    assert len(none) == 1 and "No measured data or reference values loaded for Xe-136" in none[0]
    pb = "\n".join(cp.lines(Nuclide.parse("Pb-208")))
    assert "listed as a stable nuclide" in pb


def test_energy_outside_range_is_refused_not_extrapolated():
    text = "\n".join(cp.data_lines("U-235", 1e9))
    assert "No extrapolation" in text or "unsupported" in text


def test_ions_do_not_change_the_comparison():
    assert cp.lines(Nuclide.parse("U-235")) == cp.lines(Nuclide.parse("U-235 3+"))


def test_explorer_status_has_data_block_and_energy_presets():
    m = ex.ExplorerModel()
    m.set_from_text("Ag-109")
    assert "(n,gamma)" in "\n".join(m.status_lines())
    assert m.set_energy_label("14 MeV") and m.energy_eV == 14e6 and "at 14 MeV" in "\n".join(m.status_lines())
    assert not m.set_energy_label("nonsense") and m.energy_eV == 14e6


def test_window_energy_selector_updates_model():
    src = Path(__file__).resolve().parents[1] / "src" / "neutronsim" / "explorer.py"
    with stubbed_tk():
        mod = load_source("neutronsim.explorer_stubbed2", src)
        try:
            app = mod.ExplorerApp(mod.ExplorerModel())
            app.model.set_from_text("U-235")
            app.energy_var.set("1 MeV")
            assert app.model.energy_eV == 1e6 and "at 1 MeV" in app.status_var.get()
        finally:
            sys.modules.pop("neutronsim.explorer_stubbed2", None)


def test_cli_compare(capsys):
    assert cli.main(["compare", "B-10"]) == 0
    out = capsys.readouterr().out
    assert "MODEL ESTIMATE" in out and "DATA (ENDF/B-VIII.0" in out and "MODEL ONLY" in out
    assert cli.main(["compare", "U-235", "--energy", "14", "--unit", "MeV"]) == 0 and "at 14 MeV" in capsys.readouterr().out
    assert cli.main(["compare", "C-14"]) == 0 and "REFERENCE" in capsys.readouterr().out
    assert cli.main(["compare", "nonsense"]) == 2
