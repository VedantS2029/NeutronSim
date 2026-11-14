"""Visualizer: one neutron meets one nucleus, drawn as protons and neutrons.

What is real and what is illustration
  * The OUTCOME of each fired neutron is drawn at random from the evaluated-data probabilities
    (core.sample_outcomes) - the same numbers the wizard and CLI print.
  * Nucleon counts come from the isotope name (Z from the element symbol, A from the mass number).
  * Residual nuclei are plain nucleon bookkeeping (Z and A are conserved). That is arithmetic, not
    data; the evaluated files read by this program do not contain reaction products.
  * Positions, speeds, angles and colours are SCHEMATIC. Nothing here predicts what a real neutron does.
  * Fission is shown only as 'the nucleus splits'. Fragments, emitted neutrons, energy release and
    chain reactions are not modelled (out of scope by design).

All drawing is described as plain primitives by scene()/tally_scene() so it can be tested without a
display; the tkinter part only paints those primitives.
"""
from __future__ import annotations

import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

from . import channels, core
from .scaling import SCREEN_FRACTION, TASKBAR_ALLOWANCE, fit_scale, fit_window, screen_size

try:  # tkinter is optional at import time (headless installs, tests)
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import ttk
except ImportError:  # pragma: no cover
    tk = None

W, H = 640, 360            # animation canvas
TW = 340                   # tally canvas width
GAP = 8                    # space between the two canvases
MIN_SCALE, MAX_SCALE = 0.2, 4.0   # how far the panel may shrink / grow from its logical W x H size
CX, CY = 400, 172          # nucleus centre
START_X = 24
T_CONTACT = 1.6            # seconds until the neutron reaches the nucleus
T_ABSORB = 0.45            # seconds to be taken up by the nucleus
BG = "#0f1724"
P_FILL, P_LINE = "#e5534b", "#7a1f1a"      # proton
N_FILL, N_LINE = "#7c9bdc", "#2d4a8a"      # neutron
GOLDEN = math.pi * (3 - math.sqrt(5))

ELEMENTS = ("n H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se "
            "Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho "
            "Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm "
            "Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()

# key: (Z, A, display label)
SPECIES = {"n": (0, 1, "n"), "p": (1, 1, "p"), "d": (1, 2, "d"), "t": (1, 3, "t"),
           "3He": (2, 3, "\u00b3He"), "a": (2, 4, "\u03b1")}

# Particles that leave the compound nucleus, by ENDF MT number (names follow channels.NAMES).
EMISSIONS: dict[int, list[str]] = {
    11: ["n", "n", "d"], 16: ["n", "n"], 17: ["n", "n", "n"], 22: ["n", "a"], 23: ["n", "a", "a", "a"],
    24: ["n", "n", "a"], 25: ["n", "n", "n", "a"], 28: ["n", "p"], 29: ["n", "a", "a"],
    30: ["n", "n", "a", "a"], 32: ["n", "d"], 33: ["n", "t"], 34: ["n", "3He"],
    35: ["n", "d", "a", "a"], 36: ["n", "t", "a", "a"], 37: ["n", "n", "n", "n"], 41: ["n", "n", "p"],
    42: ["n", "n", "n", "p"], 44: ["n", "p", "p"], 45: ["n", "p", "a"],
    103: ["p"], 104: ["d"], 105: ["t"], 106: ["3He"], 107: ["a"], 108: ["a", "a"], 109: ["a", "a", "a"],
    111: ["p", "p"], 112: ["p", "a"], 113: ["t", "a", "a"], 114: ["d", "a", "a"],
    115: ["p", "d"], 116: ["p", "t"], 117: ["d", "a"],
}
_LOCAL = {1: [(0, 0)], 2: [(-1.0, 0), (1.0, 0)], 3: [(-1.0, 0.6), (1.0, 0.6), (0, -1.1)],
          4: [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]}   # in nucleon radii: no overlap


# --------------------------------------------------------------------------- bookkeeping
def parse_isotope(iso: str) -> tuple[str, int, int]:
    m = re.fullmatch(r"([A-Z][a-z]?)-(\d+)", iso)
    if not m or m.group(1) not in ELEMENTS:
        raise core.InputError(f"cannot draw isotope {iso!r}")
    sym, a = m.group(1), int(m.group(2))
    return sym, ELEMENTS.index(sym), a


def nuclide(z: int, a: int) -> str:
    return f"{ELEMENTS[z]}-{a}"


def bookkeeping(z: int, a: int, mt: int | None) -> tuple[str, list[str], tuple[int, int] | None]:
    """(kind, emitted species, residual (Z, A)). Conserves Z and A (compound has A+1 nucleons)."""
    if mt == 2:
        return "elastic", [], (z, a)
    if mt == 4:
        return "inelastic", [], (z, a)
    if mt == 18:
        return "fission", [], None
    if mt == 102:
        return "capture", [], (z, a + 1)
    if mt in EMISSIONS:
        em = EMISSIONS[mt]
        zr = z - sum(SPECIES[s][0] for s in em)
        ar = a + 1 - sum(SPECIES[s][1] for s in em)
        n_have = a + 1 - z
        n_need = sum(SPECIES[s][1] - SPECIES[s][0] for s in em)
        p_need = sum(SPECIES[s][0] for s in em)
        if zr >= 1 and ar >= 1 and p_need <= z and n_need <= n_have:
            return "emit", list(em), (zr, ar)
    return "other", [], None


# --------------------------------------------------------------------------- events
@dataclass
class Nuc:
    Z: int
    A: int
    R: float
    rn: float
    types: list[str]                       # length A+1; last is the incoming neutron
    pos: list[tuple[float, float]]         # length A+1, relative to the nucleus centre
    px: float                              # contact point, relative
    py: float


def make_nucleus(z: int, a: int, impact: float) -> Nuc:
    big_r = 20 + 17 * a ** (1 / 3)
    c = big_r / math.sqrt(a)
    rn = max(3.0, 0.85 * c)
    rng = random.Random(z * 1000 + a)
    types = ["p"] * z + ["n"] * (a - z)
    rng.shuffle(types)
    types.append("n")
    pos = [(c * math.sqrt(i + .5) * math.cos(i * GOLDEN), c * math.sqrt(i + .5) * math.sin(i * GOLDEN))
           for i in range(a)]
    rv = big_r + rn
    px = -math.sqrt(max(rv * rv - impact * impact, 0.0))
    pos.append((0.85 * px, 0.85 * impact))
    return Nuc(z, a, rv, rn, types, pos, px, impact)


@dataclass
class Event:
    iso: str
    mt: int | None
    key: str
    title: str
    kind: str                              # idle elastic inelastic capture emit fission other
    nuc: Nuc
    angle: float = 0.0                     # scattering direction (screen radians)
    gamma_angle: float = 0.0
    groups: list[dict] = field(default_factory=list)
    residual: tuple[int, int] | None = None
    equation: str = ""
    compound: str = ""
    t_event: float = 0.0
    duration: float = 0.0


def idle_event(iso: str) -> Event:
    _, z, a = parse_isotope(iso)
    return Event(iso, None, "", "", "idle", make_nucleus(z, a, 0.0))


def build_event(iso: str, mt: int | None, key: str, rng: random.Random) -> Event:
    _, z, a = parse_isotope(iso)
    kind, emitted, residual = bookkeeping(z, a, mt) if mt is not None else ("other", [], None)
    rv = 20 + 17 * a ** (1 / 3) + max(3.0, 0.85 * (20 + 17 * a ** (1 / 3)) / math.sqrt(a))
    impact = rng.uniform(-0.55, 0.55) * rv
    nuc = make_nucleus(z, a, impact)
    sgn = 1 if impact >= 0 else -1
    ev = Event(iso, mt, key, channels.name(mt) if mt is not None else "Outcome not attributed to a listed channel",
               kind, nuc, angle=sgn * math.radians(rng.uniform(35, 140)), gamma_angle=rng.uniform(-0.9 * math.pi, -0.1 * math.pi),
               residual=residual, compound=nuclide(z, a + 1))
    ta = T_CONTACT + T_ABSORB
    if kind == "elastic":
        ev.equation, ev.t_event, ev.duration = f"{iso} + n \u2192 {iso} + n", T_CONTACT, T_CONTACT + 2.6
    elif kind == "inelastic":
        ev.equation = f"{iso} + n \u2192 {iso}* + n\u2032 \u2192 {iso} + n\u2032 + \u03b3"
        ev.t_event, ev.duration = T_CONTACT + 1.0, T_CONTACT + 3.4
    elif kind == "capture":
        ev.equation = f"{iso} + n \u2192 {ev.compound}* \u2192 {nuclide(*residual)} + \u03b3"
        ev.t_event = ta + 1.0
        ev.duration = ev.t_event + 2.0
    elif kind == "emit":
        labels = " + ".join(SPECIES[s][2] for s in emitted)
        ev.equation = f"{iso} + n \u2192 {ev.compound}* \u2192 {nuclide(*residual)} + {labels}"
        ev.t_event, ev.duration = ta + 1.5, ta + 1.5 + 2.4
        types = nuc.types
        pool_p = [i for i in range(a, -1, -1) if types[i] == "p"]
        pool_n = [i for i in range(a, -1, -1) if types[i] == "n"]
        base = rng.uniform(0, 2 * math.pi)
        for k, sp in enumerate(emitted):
            zs, as_, lab = SPECIES[sp]
            members = [(pool_n.pop(0), "n") for _ in range(as_ - zs)] + [(pool_p.pop(0), "p") for _ in range(zs)]
            ev.groups.append({"sp": sp, "label": lab, "members": members,
                              "dir": base + k * 2 * math.pi / len(emitted)})
    elif kind == "fission":
        ev.equation = f"{iso} + n \u2192 {ev.compound}* \u2192 two fragments"
        ev.t_event, ev.duration = ta + 1.2, ta + 1.2 + 2.2
    else:
        ev.equation = "Products are not shown for this channel."
        ev.t_event, ev.duration = ta + 1.2, ta + 1.2 + 1.4
    return ev


def outcome_table(res: core.Result) -> list[tuple[str, float]]:
    """(label, normalised probability) rows, built exactly as core.sample_outcomes builds its categories."""
    cats = [(f"MT{c.mt} {c.name}", c.probability) for c in res.channels if c.probability]
    if res.remainder_fraction is not None and abs(res.remainder_fraction) > core.REMAINDER_REL_TOL:
        cats.append(("unattributed remainder", res.remainder_fraction))
    tot = sum(p for _, p in cats) or 1.0
    return [(k, p / tot) for k, p in cats]


def fire(res: core.Result, rng: random.Random) -> Event:
    """Draw ONE outcome from the model probabilities and build its animation."""
    seed = rng.getrandbits(31)
    rows = core.sample_outcomes(res, 1, seed)["rows"]
    key = next(r["outcome"] for r in rows if r["count"] == 1)
    m = re.match(r"MT(\d+) ", key)
    return build_event(res.isotope, int(m.group(1)) if m else None, key, random.Random(seed ^ 0x9E3779B9))


# --------------------------------------------------------------------------- scene (pure)
def next_isotope(ev: Event, start_iso: str, persist: bool) -> tuple[str, str]:
    """(isotope to show after this shot, note). With persist (the default) the nucleus stays where the shot left it: the capture product, the
    residual after particle emission, or the same nucleus after scattering. Fission destroys it, so the start isotope is restored; a channel
    whose products are not attributed (MT unknown) leaves the nucleus as it was."""
    if not persist:
        return start_iso, ""
    if ev.kind == "fission":
        return start_iso, f"fission destroyed {ev.iso}: restarted from {start_iso}"
    if ev.residual is None:
        return ev.iso, "products not attributed: nucleus left as it was"
    new = nuclide(*ev.residual)
    return new, (f"{ev.iso} \u2192 {new}" if new != ev.iso else "")


def default_resolver(energy_eV: float):
    """iso -> core.Result at this energy: evaluated data where the project has it, otherwise the labelled model estimate (estsim)."""
    def resolve(iso: str) -> core.Result:
        from . import estsim, explorer
        from .nuclide import Nuclide
        if iso in explorer.data_isotopes():
            res = core.evaluate(core.Dataset(iso), energy_eV)
            if res.supported:
                return res
        return estsim.visual_result(Nuclide.parse(iso), energy_eV)
    return resolve


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _nucleon(P, x, y, typ, r):
    P.append(("circle", x, y, r, P_FILL if typ == "p" else N_FILL, P_LINE if typ == "p" else N_LINE, 1))


def _legend(P):
    P.append(("circle", W - 128, 20, 6, P_FILL, P_LINE, 1))
    P.append(("text", W - 118, 20, "proton", "#c9d6ea", 10, "w", False))
    P.append(("circle", W - 66, 20, 6, N_FILL, N_LINE, 1))
    P.append(("text", W - 56, 20, "neutron", "#c9d6ea", 10, "w", False))


def _gamma(P, ev, ts, t0, ox, oy):
    d = ev.nuc.R + 200 * (ts - t0)
    if d <= ev.nuc.R:
        return
    tail = max(ev.nuc.R, d - 80)
    a = ev.gamma_angle
    pts, dd = [], tail
    while dd <= d:
        off = 6 * math.sin(0.7 * dd - 8 * ts)
        pts += [CX + ox + math.cos(a) * dd - math.sin(a) * off, CY + oy + math.sin(a) * dd + math.cos(a) * off]
        dd += 4
    if len(pts) >= 4:
        P.append(("line", pts, "#ffd54a", 2, False))
        P.append(("text", pts[-2] + 8, pts[-1] - 8, "\u03b3", "#ffd54a", 12, "w", True))


def _captions(P, ev, ts):
    n = ev.nuc
    tc, ta, te = T_CONTACT, T_CONTACT + T_ABSORB, ev.t_event
    counts = (n.Z, n.A - n.Z)
    eq = ""
    if ev.kind == "idle":
        top = f"{ev.iso}: press \u201cFire neutron\u201d"
    elif ts < tc:
        top = f"A neutron approaches {ev.iso}"
    else:
        top = ev.title
        if ev.kind in ("elastic", "inelastic"):
            eq = ev.equation if ts > tc + 0.3 else ""
        elif ts < ta:
            pass
        elif ts < te:
            eq = f"{ev.iso} + n \u2192 {ev.compound}* (excited compound nucleus)"
            counts = (n.Z, n.A + 1 - n.Z)
        else:
            eq = ev.equation
            if ev.residual:
                counts = (ev.residual[0], ev.residual[1] - ev.residual[0])
            else:
                counts = (n.Z, n.A + 1 - n.Z)
            if ev.kind == "fission":
                counts = None
    P.append(("rect", 0, 0, W, 38, BG, BG))
    P.append(("rect", 0, H - 62, W, H, BG, BG))
    P.append(("text", 14, 14, top, "#ffffff", 15, "nw", True))
    if eq:
        P.append(("text", 14, H - 50, eq, "#cfe3ff", 13, "w", False))
    if ev.kind == "fission" and ts >= te:
        P.append(("text", 14, H - 30, "fragment identities, neutrons and energy release are not modelled",
                  "#9db4d4", 11, "w", False))
    elif counts is not None:
        P.append(("text", 14, H - 30, f"protons {counts[0]}    neutrons {counts[1]}", "#9db4d4", 11, "w", False))
    P.append(("text", 14, H - 12, "Schematic: only the outcome comes from the data; positions and motion are illustrative.",
              "#6f84a3", 9, "w", False))


def scene(ev: Event, ts: float) -> list[tuple]:
    P = _scene_body(ev, ts)
    _legend(P)                      # last, so it sits on top of the caption band
    return P


def _scene_body(ev: Event, ts: float) -> list[tuple]:
    P: list[tuple] = []
    n, a, rn = ev.nuc, ev.nuc.A, ev.nuc.rn
    tc, ta, te = T_CONTACT, T_CONTACT + T_ABSORB, ev.t_event
    contact = (CX + n.px, CY + n.py)
    slot = (CX + n.pos[a][0], CY + n.pos[a][1])
    spr = max(rn, 6.0)

    def cluster(idx, ox=0.0, oy=0.0, sx=1.0, sy=1.0, gap=0.0):
        for i in idx:
            x, y = n.pos[i]
            _nucleon(P, CX + ox + x * sx + (gap if x >= 0 else -gap), CY + oy + y * sy, n.types[i], rn)

    if ev.kind == "idle":
        cluster(range(a))
        _captions(P, ev, ts)
        return P

    absorbed = ev.kind in ("capture", "emit", "fission", "other")
    ox = oy = 0.0
    if ev.kind in ("elastic", "inelastic") and ts > tc:
        k = ts - tc
        f = math.sin(18 * k) * math.exp(-3 * k) * min(8.0, 14.0 / a ** (1 / 3))
        ox, oy = -math.cos(ev.angle) * f, -math.sin(ev.angle) * f

    if ts < tc:                                               # approach
        u = ts / tc
        ny = CY + n.py
        P.append(("line", [START_X, ny, contact[0], ny], "#33445f", 1, True))
        cluster(range(a))
        P.append(("circle", START_X + (contact[0] - START_X) * u, ny, spr, N_FILL, "#ffffff", 2))
        _captions(P, ev, ts)
        return P

    k = ts - tc
    if ev.kind in ("elastic", "inelastic"):
        cluster(range(a), ox, oy)
        speed = 220.0 if ev.kind == "elastic" else 130.0
        P.append(("circle", contact[0] + math.cos(ev.angle) * speed * k,
                  contact[1] + math.sin(ev.angle) * speed * k, spr, N_FILL, "#ffffff", 2))
        if ev.kind == "inelastic" and k < ev.t_event - tc:
            P.append(("circle", CX + ox, CY + oy, n.R + 4 + 3 * math.sin(10 * k), "", "#ffb347", 3))
        if ev.kind == "inelastic":
            _gamma(P, ev, ts, ev.t_event, ox, oy)
        _captions(P, ev, ts)
        return P

    # --- absorbing kinds
    if ts < ta:
        u = _clamp(k / T_ABSORB)
        cluster(range(a))
        P.append(("circle", contact[0] + (slot[0] - contact[0]) * u, contact[1] + (slot[1] - contact[1]) * u,
                  spr, N_FILL, "#ffffff", 2))
        _captions(P, ev, ts)
        return P

    if ev.kind == "fission":
        s = _clamp((ts - ta) / (te - ta))
        sx, sy = 1 + 0.35 * s, 1 - 0.15 * s
        shake = 2.5 * math.sin(30 * ts) * (1 - s) if s < 1 else 0.0
        gap = 0.0 if ts < te else min(6 + (ts - te) * 40, 45.0)   # stays on screen
        cluster(range(a + 1), shake, 0.0, sx, sy, gap)
        _captions(P, ev, ts)
        return P

    if ts < te:                                               # excited compound nucleus
        cluster(range(a + 1))
        P.append(("circle", CX, CY, n.R + 4 + 3 * math.sin(10 * (ts - ta)), "", "#ffb347", 3))
        _captions(P, ev, ts)
        return P

    if ev.kind == "emit":
        gone = {i for g in ev.groups for i, _ in g["members"]}
        cluster([i for i in range(a + 1) if i not in gone])
        dt = ts - te
        u = _clamp(dt / 0.35)
        for g in ev.groups:
            mem = g["members"]
            cx0 = sum(n.pos[i][0] for i, _ in mem) / len(mem)
            cy0 = sum(n.pos[i][1] for i, _ in mem) / len(mem)
            tx, ty = cx0 + math.cos(g["dir"]) * 120 * dt, cy0 + math.sin(g["dir"]) * 120 * dt
            loc = _LOCAL[min(len(mem), 4)]
            for j, (i, typ) in enumerate(mem):
                lx, ly = loc[j % len(loc)]
                x = n.pos[i][0] * (1 - u) + (tx + lx * rn) * u
                y = n.pos[i][1] * (1 - u) + (ty + ly * rn) * u
                _nucleon(P, CX + x, CY + y, typ, rn)
            P.append(("text", CX + tx, CY + ty - 2.4 * rn - 6, g["label"], "#ffffff", 12, "s", True))
    else:
        cluster(range(a + 1))
        if ev.kind == "capture":
            _gamma(P, ev, ts, te, 0.0, 0.0)
    _captions(P, ev, ts)
    return P


# --------------------------------------------------------------------------- tally (pure)
def tally_scene(table: list[tuple[str, float]], counts: Counter, total: int, w: int = TW, h: int = H) -> list[tuple]:
    P: list[tuple] = [("text", 10, 12, f"Outcome tally   (neutrons fired: {total})", "#ffffff", 13, "nw", True),
                      ("text", 10, 32, "bar = model probability    tick = share of your shots", "#9db4d4", 9, "nw", False)]
    if not table:
        return P
    row_h = min(36.0, (h - 56) / len(table))
    bw = w - 20
    for i, (key, p) in enumerate(table):
        y = 56 + i * row_h
        label = key if len(key) <= 32 else key[:31] + "\u2026"
        P.append(("text", 10, y, label, "#e6eefb", 10, "nw", False))
        P.append(("rect", 10, y + 16, 10 + bw, y + 24, "#1c2940", "#1c2940"))
        P.append(("rect", 10, y + 16, 10 + max(bw * p, 1.0), y + 24, "#4c78c9", "#4c78c9"))
        if total:
            frac = counts.get(key, 0) / total
            x = 10 + bw * frac
            P.append(("line", [x, y + 12, x, y + 28], "#ffb347", 3, False))
            P.append(("text", w - 10, y, f"{counts.get(key, 0)}/{total}   p={p:.3g}", "#ffb347", 9, "ne", False))
        else:
            P.append(("text", w - 10, y, f"p={p:.3g}", "#9db4d4", 9, "ne", False))
    return P


# --------------------------------------------------------------------------- tkinter panel
SPEEDS = {"Slow": 0.5, "Normal": 1.0, "Fast": 2.0}

if tk is not None:

    class _Painter:
        def __init__(self, family: str):
            self.family, self.fonts = family, {}

        def font(self, size: int, bold: bool):
            key = (size, bold)
            if key not in self.fonts:
                self.fonts[key] = tkfont.Font(family=self.family, size=size, weight="bold" if bold else "normal")
            return self.fonts[key]

        def paint(self, canvas, prims, k: float = 1.0):
            """Draw scene primitives (logical W x H units) onto `canvas`, everything scaled by k:
            coordinates, line widths, dashes and font sizes, so the picture keeps its proportions."""
            canvas.delete("all")
            for p in prims:
                t = p[0]
                if t == "circle":
                    _, x, y, r, fill, line, wd = p
                    canvas.create_oval((x - r) * k, (y - r) * k, (x + r) * k, (y + r) * k,
                                       fill=fill, outline=line, width=max(1, round(wd * k)))
                elif t == "line":
                    _, pts, color, wd, dash = p
                    kw = {"dash": (max(2, round(4 * k)),) * 2} if dash else {}
                    canvas.create_line(*[v * k for v in pts], fill=color, width=max(1, round(wd * k)), **kw)
                elif t == "rect":
                    _, x1, y1, x2, y2, fill, line = p
                    canvas.create_rectangle(x1 * k, y1 * k, x2 * k, y2 * k, fill=fill, outline=line)
                elif t == "text":
                    _, x, y, text, color, size, anchor, bold = p
                    canvas.create_text(x * k, y * k, text=text, fill=color,
                                       font=self.font(max(4, round(size * k)), bold), anchor=anchor)

    class VizPanel(ttk.Frame):
        """Fire neutrons one at a time (or on repeat) and watch the outcomes pile up."""

        def __init__(self, master, result: core.Result, resolver=None):
            super().__init__(master, padding=8)
            self.res = result
            self.start_res = result
            self.resolver = resolver or default_resolver(result.energy_eV)
            self.chain: list[str] = [result.isotope]
            self.by_iso: dict[str, tuple[Counter, int]] = {}
            self.table = outcome_table(result)
            self.rng = random.Random()
            self.counts: Counter = Counter()
            self.total = 0
            self.event: Event | None = None
            self.t0 = 0.0
            self._alive = True
            self.idle = idle_event(result.isotope)
            self.painter = _Painter(tkfont.nametofont("TkDefaultFont").actual("family"))

            # The two canvases always keep the scene's proportions: they are sized from the space the
            # window gives `row` (see _on_resize), centred inside it, and everything drawn is scaled by k.
            self.k = self._initial_scale()
            self.msg = tk.StringVar()
            ttk.Label(self, textvariable=self.msg).pack(side="bottom", anchor="w", pady=(6, 0))
            bar = ttk.Frame(self)
            bar.pack(side="bottom", fill="x", pady=(8, 0))
            self.fire_btn = ttk.Button(bar, text="Fire neutron", command=self.fire)
            self.fire_btn.pack(side="left")
            self.auto = tk.BooleanVar(value=False)
            ttk.Checkbutton(bar, text="Keep firing", variable=self.auto, command=self._auto_changed).pack(
                side="left", padx=10)
            ttk.Label(bar, text="Speed:").pack(side="left")
            self.speed = tk.StringVar(value="Normal")
            ttk.Combobox(bar, textvariable=self.speed, values=list(SPEEDS), width=8, state="readonly").pack(
                side="left", padx=4)
            ttk.Button(bar, text="Reset tally", command=self.reset).pack(side="right")
            ttk.Button(bar, text=f"Back to {result.isotope}", command=self.restart).pack(side="right", padx=6)
            self.persist = tk.BooleanVar(value=True)       # default: the nucleus stays where the shot left it
            ttk.Checkbutton(bar, text="Continue from the product", variable=self.persist).pack(side="right", padx=6)

            row = ttk.Frame(self, width=round((W + TW + GAP) * self.k), height=round(H * self.k))
            row.pack(side="top", fill="both", expand=True)
            row.bind("<Configure>", self._on_resize)
            holder = ttk.Frame(row)
            holder.place(relx=0.5, rely=0.5, anchor="center")
            self.canvas = tk.Canvas(holder, width=round(W * self.k), height=round(H * self.k), bg=BG,
                                    highlightthickness=0)
            self.canvas.pack(side="left")
            self.tally = tk.Canvas(holder, width=round(TW * self.k), height=round(H * self.k), bg=BG,
                                   highlightthickness=0)
            self.tally.pack(side="left", padx=(GAP, 0))

            if not self.table:      # model estimate with no predicted outcomes at this energy: show the nucleus only
                self.fire_btn.state(["disabled"])
                self.msg.set(getattr(result, "reason", "") or "No outcome probabilities are available, so nothing can be fired.")
            self.bind("<Destroy>", self._on_destroy)
            self._draw_tally()
            self._advance(0.0)
            self.after(33, self._tick)

        # --- scaling
        def _initial_scale(self) -> float:
            size = screen_size(self)
            if size is None:
                return 1.0
            sw, sh = size
            return fit_scale(sw * SCREEN_FRACTION - 40, sh * SCREEN_FRACTION - TASKBAR_ALLOWANCE - 110,
                             W + TW + GAP, H, min_k=MIN_SCALE, max_k=1.0)

        def preferred_size(self) -> tuple[int, int]:
            """Window size that shows the whole panel at its starting scale (controls bar included)."""
            return round((W + TW + GAP) * self.k) + 24, round(H * self.k) + 110

        def min_size(self) -> tuple[int, int]:
            return round((W + TW + GAP) * MIN_SCALE) + 24, round(H * MIN_SCALE) + 100

        def _on_resize(self, e):
            k = max(MIN_SCALE, min(MAX_SCALE, min((e.width - 2) / (W + TW + GAP), (e.height - 2) / H)))
            if abs(k - self.k) < 0.01:
                return
            self.k = k
            self.canvas.configure(width=round(W * k), height=round(H * k))
            self.tally.configure(width=round(TW * k), height=round(H * k))
            self._draw_tally()          # the animation canvas repaints itself on the next tick

        # --- actions
        def fire(self):
            if self.event is not None or not self._alive or not self.table:
                return
            try:
                self.event = fire(self.res, self.rng)
            except (core.UnsupportedError, core.InputError) as exc:
                self.msg.set(str(exc))
                return
            self.msg.set("")
            self.t0 = time.monotonic()

        def reset(self):
            self.counts.clear()
            self.total = 0
            self._draw_tally()

        def _show(self, iso: str, note: str = ""):
            """Switch the displayed nucleus to iso (outcome table, drawing, window title); tallies are kept per isotope."""
            if iso != self.res.isotope:
                self.by_iso[self.res.isotope] = (self.counts, self.total)
                try:
                    res = self.start_res if iso == self.start_res.isotope else self.resolver(iso)
                except Exception as exc:                  # nuclide the model cannot handle: stay where we are
                    self.msg.set(f"cannot continue with {iso} ({exc}); staying on {self.res.isotope}")
                    return
                self.res = res
                self.table = outcome_table(res)
                self.counts, self.total = self.by_iso.get(iso, (Counter(), 0))
                self.idle = idle_event(iso)
                self.chain.append(iso)
                top = self.winfo_toplevel()
                if top is not None:
                    top.title(f"Neutron visualizer - {iso} at {res.energy_eV:.6g} eV")
                if self.table:
                    self.fire_btn.state(["!disabled"])
                else:
                    self.fire_btn.state(["disabled"])
                    note = (note + "; " if note else "") + (getattr(res, "reason", "") or "no outcome probabilities, nothing can be fired")
                self._draw_tally()
            self.msg.set(note)

        def restart(self):
            if self.event is None:
                self._show(self.start_res.isotope, "back to the starting nucleus")

        def _auto_changed(self):
            if self.auto.get():
                self.fire()

        # --- animation loop
        def _on_destroy(self, e):
            if e.widget is self:
                self._alive = False

        def _tick(self):
            if not self._alive:
                return
            ts = (time.monotonic() - self.t0) * SPEEDS.get(self.speed.get(), 1.0)
            self._advance(ts)
            if self._alive:
                self.after(33, self._tick)

        def _advance(self, ts: float):
            try:
                ev = self.event
                if ev is None:
                    self.painter.paint(self.canvas, scene(self.idle, 0.0), self.k)
                    return
                if ts >= ev.duration:
                    self.painter.paint(self.canvas, scene(ev, ev.duration), self.k)
                    self.counts[ev.key] += 1
                    self.total += 1
                    self.event = None
                    self._draw_tally()
                    iso, note = next_isotope(ev, self.start_res.isotope, self.persist.get())
                    self._show(iso, note)
                    if self.auto.get():
                        self.after(350, self.fire)
                    return
                self.painter.paint(self.canvas, scene(ev, ts), self.k)
            except tk.TclError:
                self._alive = False                     # window closed mid-draw

        def _draw_tally(self):
            self.painter.paint(self.tally, tally_scene(self.table, self.counts, self.total), self.k)

    def open_window(parent, result: core.Result):
        win = tk.Toplevel(parent)
        win.title(f"Neutron visualizer - {result.isotope} at {result.energy_eV:.6g} eV")
        panel = VizPanel(win, result)
        panel.pack(fill="both", expand=True)
        fit_window(win, *panel.preferred_size(), *panel.min_size())
        return win


def main(result: core.Result) -> int:
    if tk is None:
        print("error: tkinter is not available in this Python (install python3-tk).", file=sys.stderr)
        return 2
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"error: cannot open a window ({exc}).", file=sys.stderr)
        return 2
    root.title(f"Neutron visualizer - {result.isotope} at {result.energy_eV:.6g} eV")
    panel = VizPanel(root, result)
    panel.pack(fill="both", expand=True)
    fit_window(root, *panel.preferred_size(), *panel.min_size())
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.mainloop()
    return 0
