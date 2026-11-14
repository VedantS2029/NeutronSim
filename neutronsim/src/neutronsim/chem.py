"""Step 5a: the chemistry / atomic side of a nuclide (stdlib only).

  * electron configuration: Madelung (n+l) filling order for neutral atoms, an explicit table for the ~20 neutral atoms whose
    measured ground state differs (Cr, Cu, Ag, Au, U, ...), cations lose electrons from the outermost shell first (so Fe3+ is
    3d5, not 3d3 4s2), anions add electrons in filling order. Elements 104-118 are PREDICTED (not measured) and say so.
  * mass: isotope mass from the semi-empirical model (good to ~0.01 u on 11 known masses, see check()), so a pure sample of the
    isotope has a molar mass of about that many g/mol; ion mass = atom mass - charge x electron mass (electron binding energy
    neglected: eV-keV, i.e. < 1e-4 u).
  * one-electron ions (H-like): ground-state binding energy from the Bohr formula 13.6057 eV x Z^2 and from the Dirac formula for a
    point nucleus, E = m_e c^2 (1 - sqrt(1 - (Z alpha)^2)). Both ignore nuclear size, QED and nuclear recoil.
Constants: CODATA 2018. Nothing here is a measured ionisation energy for many-electron atoms (not tabulated by this program).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import physics
from .nuclide import Nuclide, element_symbol

U_MEV = 931.49410242             # MeV per atomic mass unit (CODATA 2018)
M_E_U = 0.000548579909065        # electron mass in u
RYDBERG_EV = 13.605693122994     # Rydberg energy, eV (infinite nuclear mass)
M_E_EV = 510998.95000            # electron rest energy, eV
ALPHA = 1 / 137.035999084

ORDER = ["1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p", "6s", "4f", "5d", "6p", "7s", "5f", "6d", "7p",
         "8s", "5g", "6f", "7d", "8p"]          # beyond 7p only reached by hypothetical anions of the heaviest elements
_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
CAP = {o: 2 * (2 * _L[o[1]] + 1) for o in ORDER}
_CORES = {"He": ("1s2",), "Ne": ("[He]", "2s2", "2p6"), "Ar": ("[Ne]", "3s2", "3p6"),
          "Kr": ("[Ar]", "3d10", "4s2", "4p6"), "Xe": ("[Kr]", "4d10", "5s2", "5p6"),
          "Rn": ("[Xe]", "4f14", "5d10", "6s2", "6p6")}
_CORE_Z = {"He": 2, "Ne": 10, "Ar": 18, "Kr": 36, "Xe": 54, "Rn": 86}


def _expand(spec: str) -> dict[str, int]:
    occ: dict[str, int] = {}
    for tok in spec.split():
        if tok.startswith("["):
            for t in _CORES[tok[1:-1]]:
                occ.update(_expand(t)) if t.startswith("[") else occ.__setitem__(t[:2], int(t[2:]))
        else:
            occ[tok[:2]] = int(tok[2:])
    return occ


# neutral atoms whose measured ground state is not the plain Madelung filling (standard textbook values)
_EXCEPTIONS = {24: "[Ar] 3d5 4s1", 29: "[Ar] 3d10 4s1", 41: "[Kr] 4d4 5s1", 42: "[Kr] 4d5 5s1", 44: "[Kr] 4d7 5s1",
               45: "[Kr] 4d8 5s1", 46: "[Kr] 4d10", 47: "[Kr] 4d10 5s1", 57: "[Xe] 5d1 6s2", 58: "[Xe] 4f1 5d1 6s2",
               64: "[Xe] 4f7 5d1 6s2", 78: "[Xe] 4f14 5d9 6s1", 79: "[Xe] 4f14 5d10 6s1", 89: "[Rn] 6d1 7s2",
               90: "[Rn] 6d2 7s2", 91: "[Rn] 5f2 6d1 7s2", 92: "[Rn] 5f3 6d1 7s2", 93: "[Rn] 5f4 6d1 7s2",
               96: "[Rn] 5f7 6d1 7s2", 103: "[Rn] 5f14 7s2 7p1"}
PREDICTED_FROM_Z = 104


def _fill(n_e: int) -> dict[str, int]:
    occ, left = {}, n_e
    for o in ORDER:
        k = min(CAP[o], left)
        if k:
            occ[o] = k
        left -= k
        if not left:
            break
    return occ


def neutral_config(z: int) -> dict[str, int]:
    return _expand(_EXCEPTIONS[z]) if z in _EXCEPTIONS else _fill(z)


def config(z: int, electrons: int) -> dict[str, int]:
    """Subshell occupations for an atom/ion with `electrons` electrons around Z protons."""
    occ = dict(neutral_config(z))
    if electrons < z:                                   # cation: remove from the highest n, then highest l
        for _ in range(z - electrons):
            o = max((s for s in occ if occ[s] > 0), key=lambda s: (int(s[0]), _L[s[1]]))
            occ[o] -= 1
            if not occ[o]:
                del occ[o]
    else:                                               # anion: add in filling order where there is room
        for _ in range(electrons - z):
            o = next(s for s in ORDER if occ.get(s, 0) < CAP[s])
            occ[o] = occ.get(o, 0) + 1
    return occ


def config_string(occ: dict[str, int]) -> str:
    if not occ:
        return "(no electrons: bare nucleus)"
    key = lambda s: (int(s[0]), _L[s[1]])
    core = ""
    for name in ("Rn", "Xe", "Kr", "Ar", "Ne", "He"):  # largest complete noble-gas core present
        full = _expand(f"[{name}]" if name != "He" else "1s2")
        if all(occ.get(s) == c for s, c in full.items()):
            core, occ = f"[{name}] ", {s: c for s, c in occ.items() if s not in full}
            break
    return (core + " ".join(f"{s}{occ[s]}" for s in sorted(occ, key=key))).strip()


def mass_u(z: int, n: int) -> float:
    return physics.atomic_mass(z, n) / U_MEV


def ion_mass_u(nuc: Nuclide) -> float:
    return mass_u(nuc.Z, nuc.N) - nuc.charge * M_E_U


def hydrogenlike_binding_eV(z: int) -> tuple[float, float]:
    """(Bohr, Dirac point-nucleus) ground-state binding energy in eV of a one-electron ion with nuclear charge z."""
    za = z * ALPHA
    return RYDBERG_EV * z * z, M_E_EV * (1 - math.sqrt(1 - za * za))


def render(nuc: Nuclide) -> list[str]:
    out = ["CHEMISTRY / ATOMIC (standard formulas; mass from the model)"]
    occ = config(nuc.Z, nuc.electrons)
    tag = f"{nuc.symbol}{'' if nuc.charge == 0 else (('+' if nuc.charge > 0 else '-') if abs(nuc.charge) == 1 else f'{abs(nuc.charge)}' + ('+' if nuc.charge > 0 else '-'))}"
    out.append(f"  electron configuration of {tag}: {config_string(occ)}")
    if nuc.Z >= PREDICTED_FROM_Z:
        out.append("    (elements 104+ : configuration is a prediction, not measured)")
    elif nuc.Z in _EXCEPTIONS and nuc.charge == 0:
        out.append(f"    (neutral {element_symbol(nuc.Z)} is one of the exceptions to simple filling order)")
    if nuc.charge < 0:
        out.append("    (anion: extra electrons placed in filling order)")
    m = mass_u(nuc.Z, nuc.N)
    out.append(f"  isotope mass (model): {m:.4f} u -> a pure sample would have molar mass of about {m:.2f} g/mol "
               f"(model error up to ~0.01 u)")
    if nuc.charge != 0:
        out.append(f"  ion mass: {ion_mass_u(nuc):.4f} u (atom mass {'-' if nuc.charge > 0 else '+'} {abs(nuc.charge)} x electron mass; "
                   "electron binding energy neglected)")
    if nuc.electrons == 1:
        b, d = hydrogenlike_binding_eV(nuc.Z)
        fmt = lambda v: f"{v / 1e3:.2f} keV" if v >= 1e4 else f"{v:.2f} eV"
        out.append(f"  one-electron ion: ground-state binding {fmt(b)} (Bohr), {fmt(d)} (Dirac, point nucleus); "
                   "ignores nuclear size, QED and recoil")
    elif nuc.electrons > 1:
        out.append("  ionisation energies of many-electron atoms are not tabulated here.")
    return out


def check(ref_path: Path = physics.REFERENCE_FILE) -> tuple[list[str], dict]:
    ref = json.loads(Path(ref_path).read_text())["atomic_mass_u"]
    out, errs = ["Isotope mass, model vs reference (u; reference typed from memory):"], []
    for r in ref:
        m = mass_u(r["Z"], r["N"])
        errs.append(m - r["m"])
        out.append(f"  {r['iso']:<7} model {m:12.5f}  ref {r['m']:12.5f}  diff {m - r['m']:+.5f}")
    st = {"max_abs": max(abs(e) for e in errs), "mean_abs": sum(abs(e) for e in errs) / len(errs)}
    out.append(f"  mean |diff| {st['mean_abs']:.4f} u, max {st['max_abs']:.4f} u")
    return out, st
