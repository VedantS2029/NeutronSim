"""Step 4 of the isotope-explorer: put the physics model next to real data.

Three kinds of "real" are used, and each line says which:
  DATA       evaluated nuclear data (ENDF/B-VIII.0 files shipped with this project): neutron cross sections and the
             reaction Q-values stored in them. Only 4 isotopes (B-10, Al-27, Ag-109, U-235).
  REFERENCE  a few binding energies / decay energies typed from memory (validation/physics_reference.json), ~AME2020.
  MODEL      physics.py (semi-empirical mass formula).
Nothing is ever shown for a nuclide without data or reference values: the output says so instead of guessing.

Reaction Q-values are ground-state to ground-state and include only the particles named by the ENDF MT number;
fission (MT18) and inelastic scattering (MT4/5) have no single such Q and are not modelled.
Evaluated Q = 0 in these files means "not stored", not "zero energy release", so those channels are skipped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from . import channels, core, physics, validation
from .nuclide import MAX_Z, Nuclide

# emitted particles per ENDF MT number (excluding gamma rays); the daughter is whatever nucleons remain
EMISSIONS: dict[int, tuple[str, ...]] = {
    102: (), 16: ("n", "n"), 17: ("n",) * 3, 37: ("n",) * 4, 103: ("p",), 104: ("d",), 105: ("t",), 106: ("he3",),
    107: ("a",), 108: ("a", "a"), 109: ("a",) * 3, 111: ("p", "p"), 112: ("p", "a"), 113: ("t", "a", "a"),
    114: ("d", "a", "a"), 115: ("p", "d"), 116: ("p", "t"), 117: ("d", "a"), 22: ("n", "a"), 23: ("n",) + ("a",) * 3,
    24: ("n", "n", "a"), 25: ("n",) * 3 + ("a",), 28: ("n", "p"), 29: ("n", "a", "a"), 30: ("n", "n", "a", "a"),
    32: ("n", "d"), 33: ("n", "t"), 34: ("n", "he3"), 35: ("n", "d", "a", "a"), 36: ("n", "t", "a", "a"),
    41: ("n", "n", "p"), 42: ("n", "n", "n", "p"), 44: ("n", "p", "p"), 45: ("n", "p", "a"),
}
# binding energies of the light emitted nuclei (MeV): deuteron, triton, helium-3 (He-4 comes from physics.py)
_B_LIGHT = {"d": 2.224566, "t": 8.481798, "he3": 7.718058}
_ZN = {"n": (0, 1), "p": (1, 0), "d": (1, 1), "t": (1, 2), "he3": (2, 1), "a": (2, 2)}


def _atom_mass(p: str) -> float:
    """Neutral-atom rest energy of an emitted particle (MeV)."""
    z, n = _ZN[p]
    if p == "a":
        return physics.M_HE4
    return z * (physics.M_P + physics.M_E) + n * physics.M_N - _B_LIGHT.get(p, 0.0)


def reaction_q(z: int, n: int, mt: int) -> float | None:
    """Model Q (MeV) for neutron + (Z,N) -> daughter + emitted particles. None if the MT is not modelled or has no valid daughter."""
    if mt not in EMISSIONS:
        return None
    em = EMISSIONS[mt]
    zd, nd = z - sum(_ZN[p][0] for p in em), n + 1 - sum(_ZN[p][1] for p in em)
    if zd < 0 or nd < 0 or zd > MAX_Z or (zd == 0 and nd > 0):
        return None
    daughter = 0.0 if (zd, nd) == (0, 0) else physics.atomic_mass(zd, nd)      # (0,0): everything is emitted (e.g. B-10 -> t + 2 alpha)
    return physics.atomic_mass(z, n) + physics.M_N - daughter - sum(_atom_mass(p) for p in em)


@dataclass
class QRow:
    mt: int
    name: str
    data_MeV: float
    model_MeV: float

    @property
    def diff(self) -> float:
        return self.model_MeV - self.data_MeV


@lru_cache(maxsize=None)
def q_rows(iso: str) -> tuple[QRow, ...]:
    """Model vs evaluated Q for every channel of `iso` whose Q is stored (nonzero) and modelled."""
    ds = core.Dataset(iso)
    nuc = Nuclide.parse(iso)
    rows = []
    for mt in sorted(ds.tabs):
        q_data = ds.tabs[mt].qm / 1e6
        q_model = reaction_q(nuc.Z, nuc.N, mt)
        if q_data != 0.0 and q_model is not None:
            rows.append(QRow(mt, channels.name(mt), q_data, q_model))
    return tuple(rows)


def q_summary(rows) -> str:
    if not rows:
        return "no channel with a stored, modelled Q-value"
    mean = sum(abs(r.diff) for r in rows) / len(rows)
    worst = max(rows, key=lambda r: abs(r.diff))
    return (f"{len(rows)} channels: model differs from evaluated Q by {mean:.2f} MeV on average, "
            f"worst {worst.diff:+.2f} MeV ({worst.name})")


# --------------------------------------------------------------------------- reference points typed from memory
@lru_cache(maxsize=1)
def _reference() -> dict:
    return json.loads(physics.REFERENCE_FILE.read_text())


def reference_lines(nuc: Nuclide) -> list[str]:
    ref, out = _reference(), []
    for r in ref["binding_energy_MeV"]:
        if (r["Z"], r["N"]) == (nuc.Z, nuc.N):
            m = physics.binding_energy(nuc.Z, nuc.N)
            out.append(f"  binding energy: model {m:.1f} MeV vs reference {r['B']:.1f} MeV (diff {m - r['B']:+.1f})")
    for r in ref["decay_Q_MeV"]:
        if (r["Z"], r["N"]) == (nuc.Z, nuc.N):
            key = {"beta-": "beta-", "EC": "EC", "alpha": "alpha"}[r["mode"]]
            m = physics.decay_q(nuc.Z, nuc.N)[key]
            out.append(f"  real {r['mode']} decay Q: model {m:+.2f} MeV vs reference {r['Q']:+.2f} MeV (diff {m - r['Q']:+.2f})"
                       + ("" if m > 0 else "  <- model says NOT allowed, but this decay is real"))
    if nuc.key in [f"{n}" for n in ref["stable_nuclides"]]:
        out.append(f"  {nuc.key} is listed as a stable nuclide; model verdict: {physics.estimate_nuclide(nuc).verdict}")
    return out


# --------------------------------------------------------------------------- measured neutron data
ENERGY_PRESETS = [("Thermal (0.0253 eV)", 0.0253), ("1 keV", 1e3), ("1 MeV", 1e6), ("14 MeV", 14e6)]


def _fmt_e(e: float) -> str:
    return f"{e / 1e6:g} MeV" if e >= 1e6 else (f"{e / 1e3:g} keV" if e >= 1e3 else f"{e:g} eV")


def data_lines(iso: str, e_eV: float, full: bool = False) -> list[str]:
    ds = core.Dataset(iso)
    res = core.evaluate(ds, e_eV)
    out = [f"DATA ({ds.library} evaluated data; real measurements/evaluations, not a model)"]
    if not res.supported:
        out.append(f"  at {_fmt_e(e_eV)}: {res.reason}")
    else:
        rows = [c for c in res.channels if c.sigma_barn]
        rows.sort(key=lambda c: -(c.probability or 0))
        out.append(f"  at {_fmt_e(e_eV)} a neutron meeting {iso} (total cross section {res.total.sigma_barn:.6g} barn):")
        for c in rows[: (len(rows) if full else 3)]:
            out.append(f"    {c.name:<46}{c.sigma_barn:>12.6g} b   p = {c.probability:.4g}")
        if not full and len(rows) > 3:
            out.append(f"    ... {len(rows) - 3} smaller channels (see 'neutronsim xs {iso} --energy ...')")
        out.append("  (probabilities are shares of the cross section: a calculation, not a prediction for one neutron)")
    for case in validation.run():
        if case.isotope == iso:
            out.append(f"  thermal absorption: evaluated {case.evaluated_b:.5g} b vs Sears 1992 {case.reference_b:g} +/- {case.reference_u_b:g} b "
                       f"-> {case.status} ({case.sigma_units:+.1f} sigma)")
    rows_q = q_rows(iso)
    out.append("  model vs evaluated reaction Q-values: " + q_summary(rows_q))
    if full and rows_q:
        out.append(f"    {'MT':<5}{'channel':<46}{'evaluated':>10}{'model':>9}{'diff':>8}  (MeV)")
        for r in rows_q:
            out.append(f"    {r.mt:<5}{r.name:<46}{r.data_MeV:>+10.3f}{r.model_MeV:>+9.3f}{r.diff:>+8.2f}")
        out.append("    Channels with Q stored as 0 in the file are skipped above: 0 means 'not given'.")
    if full:
        nuc = Nuclide.parse(iso)
        extra = [(mt, reaction_q(nuc.Z, nuc.N, mt)) for mt in ds.partial_mts
                 if ds.tabs[mt].qm == 0.0 and reaction_q(nuc.Z, nuc.N, mt) is not None]
        if extra:
            out.append("  MODEL ONLY (these channels have cross sections in the data but no Q stored, so nothing to compare with):")
            for mt, q in extra:
                out.append(f"    {mt:<5}{channels.name(mt):<46}{q:>+10.3f} MeV (model, uncertain by ~1-3 MeV)")
    if full:
        out.append("    Fission (MT18) and inelastic scattering (MT4/5) are not modelled.")
    return out


def lines(nuc: Nuclide, e_eV: float = 0.0253, full: bool = False) -> list[str]:
    out: list[str] = []
    try:
        has_data = nuc.key in json.loads((core.DATA_DIR / "MANIFEST.json").read_text())["isotopes"]
    except (OSError, ValueError, KeyError):
        has_data = False
    if has_data:
        out += data_lines(nuc.key, e_eV, full)
    ref = reference_lines(nuc)
    if ref:
        out += ["REFERENCE (values typed from memory, ~AME2020; verify before citing)"] + ref
    if not out:
        out = [f"No measured data or reference values loaded for {nuc.key}: the model estimate above is all there is."]
    return out
