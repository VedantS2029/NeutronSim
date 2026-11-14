"""Step 3 of the isotope-explorer: a rough physics model for ANY nuclide (stdlib only, no data files).

Model: the semi-empirical (Bethe-Weizsacker / "liquid-drop") mass formula
    B(Z,N) = aV*A - aS*A^(2/3) - aC*Z(Z-1)/A^(1/3) - aA*(A-2Z)^2/A + delta
with the common textbook coefficients aV=15.8, aS=18.3, aC=0.714, aA=23.2, aP=12 MeV and
delta = +aP/sqrt(A) (even Z, even N), -aP/sqrt(A) (odd Z, odd N), 0 otherwise.
From B we get atomic masses and from mass differences the energy release Q of each decay channel.

HONESTY NOTES (measured in validation/physics_reference.json, see `check()`):
  * Binding energies are good to ~1% for medium/heavy nuclei, worse for light ones (He-4 is off by 6 MeV of 28).
  * Q-values are only good to a few MeV (largest error on the reference set ~2.7 MeV). Many real decays have
    Q below that (C-14: 0.16 MeV), so the model cannot reliably say whether those are allowed. Such channels are
    flagged MARGINAL and the verdict says "borderline" instead of guessing.
  * Q > 0 means a channel is ENERGETICALLY ALLOWED. It says nothing about speed: half-lives, selection rules,
    shell structure, spontaneous fission and gamma emission are NOT modelled.
  * Electrons are ignored (nuclear masses are taken from neutral-atom masses; ionisation energies are eV, not MeV).
Constants: CODATA 2018 (proton/neutron/electron rest energies); He-4 binding energy 28.29567 MeV (alpha-particle mass).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .nuclide import MAX_Z, Nuclide, element_symbol

M_N, M_P, M_E = 939.56542052, 938.27208816, 0.51099895      # MeV/c^2 (CODATA 2018)
A_V, A_S, A_C, A_A, A_P = 15.8, 18.3, 0.714, 23.2, 12.0     # MeV
B_HE4 = 28.29567                                            # MeV, alpha particle
Q_UNCERTAINTY = 3.0      # MeV; channels with |Q| below this are called marginal (observed worst error ~2.7 MeV)
LIGHT_A = 20             # below this the liquid-drop picture is poor (no shell/cluster structure)
REFERENCE_FILE = Path(__file__).resolve().parents[2] / "validation" / "physics_reference.json"


def binding_energy(z: int, n: int) -> float:
    """Total binding energy in MeV (semi-empirical). A = 1 is exactly 0 by definition."""
    a = z + n
    if z < 0 or n < 0 or a < 1:
        raise ValueError(f"no nucleus with Z={z}, N={n}")
    if a == 1:
        return 0.0
    pair = A_P / math.sqrt(a) if (z % 2 == 0 and n % 2 == 0) else (-A_P / math.sqrt(a) if (z % 2 and n % 2) else 0.0)
    return A_V * a - A_S * a ** (2 / 3) - A_C * z * (z - 1) / a ** (1 / 3) - A_A * (a - 2 * z) ** 2 / a + pair


def atomic_mass(z: int, n: int) -> float:
    """Neutral-atom rest energy in MeV (electron binding neglected)."""
    return z * (M_P + M_E) + n * M_N - binding_energy(z, n)


M_HE4 = 2 * (M_P + M_E) + 2 * M_N - B_HE4
CHANNELS = ("alpha", "beta-", "beta+", "EC", "n emission", "p emission")


def _ok(z: int, n: int) -> bool:
    return 1 <= z <= MAX_Z and n >= 0 and z + n >= 1


def decay_q(z: int, n: int) -> dict[str, float | None]:
    """Energy release Q (MeV) per channel, positive = energetically allowed. None = channel impossible (no valid daughter)."""
    m = atomic_mass(z, n)
    q: dict[str, float | None] = {c: None for c in CHANNELS}
    if _ok(z - 2, n - 2) and n >= 2 and z >= 3:
        q["alpha"] = m - atomic_mass(z - 2, n - 2) - M_HE4
    if _ok(z + 1, n - 1) and n >= 1:
        q["beta-"] = m - atomic_mass(z + 1, n - 1)
    if _ok(z - 1, n + 1) and z >= 2:
        q["EC"] = m - atomic_mass(z - 1, n + 1)
        q["beta+"] = q["EC"] - 2 * M_E
    if _ok(z, n - 1) and n >= 1 and z + n >= 2:
        q["n emission"] = m - atomic_mass(z, n - 1) - M_N
    if _ok(z - 1, n) and z >= 2:
        q["p emission"] = m - atomic_mass(z - 1, n) - (M_P + M_E)
    return q


def lowest_mass_isobar(a: int) -> int:
    """Z that minimises the model's atomic mass for fixed A (the model's 'valley of stability')."""
    return min(range(1, min(a, MAX_Z) + 1), key=lambda z: atomic_mass(z, a - z))


def separation_energies(z: int, n: int) -> tuple[float | None, float | None]:
    """(S_n, S_p) in MeV: energy needed to remove one neutron / proton. None if that nucleus doesn't exist."""
    sn = binding_energy(z, n) - binding_energy(z, n - 1) if n >= 1 and z + n >= 2 else None
    sp = binding_energy(z, n) - binding_energy(z - 1, n) if z >= 2 else None
    return sn, sp


@dataclass
class Estimate:
    z: int
    n: int
    binding: float
    per_nucleon: float
    s_n: float | None
    s_p: float | None
    q: dict
    isobar_z: int
    verdict: str                      # "stable", "borderline", "unstable", "particle-unbound"
    verdict_text: str
    allowed: list = field(default_factory=list)      # [(channel, Q, marginal)], largest Q first
    notes: list = field(default_factory=list)


def estimate(z: int, n: int) -> Estimate:
    b = binding_energy(z, n)
    a = z + n
    sn, sp = separation_energies(z, n)
    q = decay_q(z, n)
    allowed = sorted(((c, v, v < Q_UNCERTAINTY) for c, v in q.items() if v is not None and v > 0), key=lambda t: -t[1])
    near = [c for c, v in q.items() if v is not None and -Q_UNCERTAINTY < v <= 0]
    unbound = [c for c, v, m in allowed if c in ("n emission", "p emission") and not m]
    firm = [c for c, v, m in allowed if not m]
    soft = [c for c, v, m in allowed if m]
    extra = f" (also allowed but marginal: {', '.join(soft)})" if soft and firm else ""
    if all(v is None for v in q.values()):
        verdict, text = "stable", "no decay products are possible for a nucleus this small"
    elif unbound:
        verdict, text = "particle-unbound", (f"model says it can shed a {'neutron' if 'n emission' in unbound else 'proton'} "
                                              "directly (beyond the drip line), so it would not hold together for long" + extra)
    elif firm:
        verdict, text = "unstable", f"energetically free to decay by {', '.join(firm)} (Q well above the model's error)" + extra
    elif allowed:
        verdict, text = "borderline", ("decay by " + ", ".join(c for c, _, _ in allowed) +
                                       " is allowed by only a small margin: within the model's error, so it cannot say")
    elif near:
        verdict, text = "borderline", ("no channel allowed, but " + ", ".join(near) +
                                       " is within the model's error of being allowed: cannot rule it out")
    else:
        verdict, text = "stable", "no decay channel is energetically allowed, by a margin larger than the model's error"
    notes = []
    if a < LIGHT_A:
        notes.append(f"A < {LIGHT_A}: the liquid-drop model is poor for light nuclei (no shell/cluster structure); treat numbers as rough.")
    if a == 1:
        notes.append("A single nucleon has no binding energy by definition.")
    return Estimate(z, n, b, b / a, sn, sp, q, lowest_mass_isobar(a), verdict, text, allowed, notes)


def render(est: Estimate) -> list[str]:
    f = lambda v: "n/a" if v is None else f"{v:+.2f}"
    lines = ["MODEL ESTIMATE (semi-empirical mass formula; approximate, not measured data)",
             f"  binding energy {est.binding:.1f} MeV = {est.per_nucleon:.2f} MeV per nucleon;"
             f" S_n {f(est.s_n)}, S_p {f(est.s_p)} MeV",
             f"  model's lowest-mass isobar for A={est.z + est.n}: Z={est.isobar_z} ({element_symbol(est.isobar_z)})",
             f"  decay energy Q per channel (MeV, + = energetically allowed; model error up to ~{Q_UNCERTAINTY:g}):"]
    for c in CHANNELS:
        v = est.q[c]
        tag = "" if v is None or v <= 0 else ("  <- allowed, marginal" if v < Q_UNCERTAINTY else "  <- allowed")
        lines.append(f"    {c:<11} {'n/a (no daughter)' if v is None else format(v, '+.2f')}{tag}")
    lines.append(f"  model verdict: {est.verdict.upper()} - {est.verdict_text}")
    lines += [f"  note: {t}" for t in est.notes]
    lines.append("  Not modelled: half-lives, spontaneous fission, gamma emission, shell effects, ionisation energies.")
    return lines


def estimate_nuclide(nuc: Nuclide) -> Estimate:
    """Electrons are ignored: the nucleus alone determines these numbers."""
    return estimate(nuc.Z, nuc.N)


# --------------------------------------------------------------------------- how wrong is it? (checked against reference points)
def check(ref_path: Path = REFERENCE_FILE) -> tuple[list[str], dict]:
    ref = json.loads(Path(ref_path).read_text())
    out, stats = [ref["note"], "", "Binding energy, model vs reference (MeV):"], {}
    errs = []
    for r in ref["binding_energy_MeV"]:
        m = binding_energy(r["Z"], r["N"])
        errs.append(m - r["B"])
        out.append(f"  {r['iso']:<7} model {m:8.1f}  ref {r['B']:8.1f}  diff {m - r['B']:+6.1f}  ({(m - r['B']) / r['B'] * 100:+.1f}%)")
    stats["B_max_abs_err"] = max(abs(e) for e in errs)
    out += ["", "Decay energy Q, model vs reference (MeV):"]
    qerrs, sign_ok = [], 0
    for r in ref["decay_Q_MeV"]:
        key = {"beta-": "beta-", "EC": "EC", "alpha": "alpha"}[r["mode"]]
        m = decay_q(r["Z"], r["N"])[key]
        qerrs.append(m - r["Q"])
        sign_ok += (m > 0) == (r["Q"] > 0)
        out.append(f"  {r['iso']:<7} {r['mode']:<6} model {m:+6.2f}  ref {r['Q']:+6.2f}  diff {m - r['Q']:+5.2f}"
                   f"{'' if m > 0 else '   <- model says NOT allowed (real decay)'}")
    stats.update(Q_max_abs_err=max(abs(e) for e in qerrs), Q_mean_abs_err=sum(abs(e) for e in qerrs) / len(qerrs),
                 Q_sign_correct=sign_ok, Q_total=len(qerrs))
    out += ["", "Nuclides that are really stable - model verdict:"]
    wrong = 0
    for iso in ref["stable_nuclides"]:
        nuc = Nuclide.parse(iso)
        e = estimate_nuclide(nuc)
        wrong += e.verdict in ("unstable", "particle-unbound")
        out.append(f"  {iso:<7} {e.verdict}")
    stats["stable_called_unstable"] = wrong
    out += ["", f"Summary: B off by up to {stats['B_max_abs_err']:.1f} MeV; Q off by {stats['Q_mean_abs_err']:.2f} MeV on average, up to "
            f"{stats['Q_max_abs_err']:.2f} MeV; real decays whose allowed/forbidden sign the model got right: {sign_ok}/{len(qerrs)}; "
            f"stable nuclides wrongly called unstable: {wrong}/{len(ref['stable_nuclides'])}."]
    return out, stats
