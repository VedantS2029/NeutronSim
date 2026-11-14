"""Box visualizer: watch neutrons move through the block, collision by collision.

What is real and what is illustration (same convention as viz.py)
  * Every path drawn here IS a history from box.run_box: the same random draws, cross sections (evaluated data, or the
    labelled physics-law model estimate for isotopes without data), collision types, energies and positions that produced
    the statistics next to it. Nothing is re-simulated for the picture.
  * The ATOMS: the engine does not place individual nuclei (at solid density there are ~1e22 per cm3, far too many to
    store or draw). What it does produce are the places where a nucleus was actually struck. Those are drawn, as small
    rings coloured by what happened there, and they stay on screen. There are NO invented background dots. The panel on the
    right gives the true atom numbers (atoms per cm3, atoms in the box, spacing, mean free path in atomic spacings).
  * Only the first N released neutrons (and the neutrons their (n,2n)-type reactions create) are drawn; the statistics
    always cover all of them.
  * SCHEMATIC: the animation clock. Every straight segment between two collisions takes the same time on screen, however
    long it really took (real times differ by many orders of magnitude; the caption gives the real time and the real path
    length so far). The 3-D view is an orthographic projection you can rotate.
  * Not drawn: gammas, charged particles, recoils (they stop at the collision point in the model). A flash marks where
    a reaction happened.

All drawing is described as plain primitives by scene()/tally_scene(), testable without a display; the tkinter part only
paints them.
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter

from . import box, channels

try:  # tkinter is optional at import time (headless installs, tests)
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import ttk
except ImportError:  # pragma: no cover
    tk = None

W, H = 640, 440                 # animation canvas (logical units)
TW = 300                        # tally canvas width
GAP = 8                         # space between the two canvases
BG = "#0f1724"
EDGE_BACK, EDGE_FRONT = "#2c3f63", "#6f8fc4"
KIND_COLOR = {"elastic": "#7fb0ff", "absorb": "#ff5c5c", "react": "#ffb347", "leak": "#4fd18b", "alive": "#9db4d4", "cap": "#9db4d4"}
KIND_TEXT = {"elastic": "elastic scattering", "react": "reaction (neutron comes out)", "absorb": "absorbed",
             "leak": "left the box", "alive": "still moving when the clock stopped", "cap": "stopped by the collision cap"}
SPEEDS = {"Slow": 2.0, "Normal": 5.0, "Fast": 12.0}          # animation steps (segments) per second
FLASH = 0.8                     # how long (in steps) a collision flash lasts
TRAIL = 3                       # segments of trail behind a moving neutron
DEFAULT_TRACE = 60


# --------------------------------------------------------------------------- geometry
def project(x: float, y: float, z: float, yaw: float, pitch: float, half: float, k: float = 1.0) -> tuple[float, float, float]:
    """Orthographic view of a point inside the cube [-half, half]^3. Returns canvas (sx, sy) and depth (larger = nearer)."""
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x1, z1 = x * cy_ + z * sy_, -x * sy_ + z * cy_            # turn about the vertical axis
    y2, z2 = y * cp - z1 * sp, y * sp + z1 * cp               # tilt
    s = k * 0.40 * min(W, H) / (half * math.sqrt(3.0))        # the cube's body diagonal fits the canvas
    return W / 2.0 + x1 * s, H / 2.0 - y2 * s, z2


def cube_corners(half: float) -> list[tuple[float, float, float]]:
    return [(sx * half, sy * half, sz * half) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


CUBE_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]   # 12 edges


def energy_color(e_eV: float) -> str:
    """Slow (thermal) = blue ... fast (MeV) = red. Colour only; not a measurement."""
    if e_eV <= 0:
        return "#9db4d4"
    f = (math.log10(max(e_eV, 1e-3)) + 2.0) / 9.0            # 0.01 eV -> 0, 10 MeV -> ~1
    f = max(0.0, min(1.0, f))
    stops = [(0.0, (76, 155, 232)), (0.5, (120, 220, 200)), (0.75, (255, 200, 80)), (1.0, (255, 92, 92))]
    for (f0, c0), (f1, c1) in zip(stops, stops[1:]):
        if f <= f1:
            t = (f - f0) / (f1 - f0)
            return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))
    return "#ff5c5c"


def struck_sites(traces: list, phase: float) -> list[tuple[float, float, float, str]]:
    """Where a nucleus has been struck by animation time `phase`: (x, y, z, kind) for every finished collision
    (kind = elastic | react | absorb). These are real sites from the run, not decoration."""
    out = []
    for tr in traces:
        for j, st in enumerate(tr.steps):
            if tr.start_phase + j + 1 > phase:
                break
            if st[5] in ("elastic", "react", "absorb"):
                out.append((st[0], st[1], st[2], st[5]))
    return out


def scale_facts(res: box.BoxResult) -> dict:
    """True length scales of this box (all from the run's own numbers; nuclear radius is the usual 1.2 A^(1/3) fm rule of thumb)."""
    from .nuclide import Nuclide
    n = res.atoms_per_cm3
    spacing = n ** (-1.0 / 3.0)
    A = Nuclide.parse(res.isotope).A
    r_nuc = 1.2e-13 * A ** (1.0 / 3.0)
    return {"atoms_per_cm3": n, "atoms_in_box": n * res.side_cm ** 3, "spacing_cm": spacing,
            "mfp_cm": res.mean_free_path_cm, "mfp_in_spacings": res.mean_free_path_cm / spacing,
            "side_in_mfp": res.side_cm / res.mean_free_path_cm, "nuclear_radius_cm": r_nuc,
            "spacing_in_nuclear_diameters": spacing / (2.0 * r_nuc)}


def path_length_cm(traces: list, phase: float) -> float:
    """Total real distance flown so far by the watched neutrons (completed segments only)."""
    tot = 0.0
    for tr in traces:
        p = (tr.x0, tr.y0, tr.z0)
        for j, st in enumerate(tr.steps):
            if tr.start_phase + j + 1 > phase:
                break
            tot += math.dist(p, st[:3])
            p = st[:3]
    return tot


def _wrap(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------- animation state
def total_phase(traces: list) -> float:
    """Animation steps needed for every recorded history to finish."""
    return float(max((t.start_phase + len(t.steps) for t in traces), default=0))


def neutron_state(tr, phase: float) -> dict | None:
    """Where one recorded history is at animation time `phase` (None before it starts)."""
    local = phase - tr.start_phase
    if local < 0:
        return None
    n = len(tr.steps)
    if n == 0:
        return {"pos": (tr.x0, tr.y0, tr.z0), "E": tr.E0, "done": tr.end or "", "age": local, "prev": None, "seg": None}
    if local >= n:
        last = tr.steps[-1]
        done = tr.end if tr.end in ("absorb", "leak", "alive", "cap") else "truncated"
        return {"pos": last[:3], "E": last[3], "done": done, "age": local - n, "prev": None, "seg": last[5]}
    i = int(local)
    frac = local - i
    p0 = (tr.x0, tr.y0, tr.z0) if i == 0 else tr.steps[i - 1][:3]
    p1 = tr.steps[i][:3]
    e0 = tr.E0 if i == 0 else tr.steps[i - 1][3]
    pos = tuple(a + (b - a) * frac for a, b in zip(p0, p1))
    return {"pos": pos, "E": e0, "done": "", "age": 0.0, "prev": p0, "seg": None,
            "flash": (tr.steps[i - 1][5], frac, p0) if i > 0 else None, "trail_from": max(0, i - TRAIL)}


def tally_at(traces: list, phase: float) -> dict:
    """What has happened among the recorded histories up to animation time `phase` (completed segments only)."""
    kinds: Counter = Counter()
    mts: Counter = Counter()
    ends: Counter = Counter()
    last = None
    for ti, tr in enumerate(traces):
        for j, st in enumerate(tr.steps):
            done_at = tr.start_phase + j + 1
            if done_at > phase:
                break
            kind = st[5]
            if kind in ("elastic", "react", "absorb"):
                kinds[kind] += 1
                mts[st[6]] += 1
            if kind in ("absorb", "leak", "alive", "cap"):
                ends[kind] += 1
            if last is None or done_at >= last[0]:
                last = (done_at, ti, st)
    return {"kinds": kinds, "mts": mts, "ends": ends, "last": last}


def caption(traces: list, phase: float) -> str:
    st = tally_at(traces, phase)["last"]
    if st is None:
        return "Neutrons start from the source position."
    _, ti, s = st
    kind = s[5]
    what = KIND_TEXT.get(kind, kind)
    if s[6] is not None and kind in ("react", "absorb"):
        what = f"MT{s[6]} {channels.name(s[6])}: {what}"
    e = s[3]
    es = f"{e / 1e6:.3g} MeV" if e >= 1e6 else f"{e / 1e3:.3g} keV" if e >= 1e3 else f"{e:.3g} eV"
    tail = "" if kind == "absorb" else f", energy now {es}"
    return f"Latest event: neutron #{ti + 1} - {what} at t = {s[4]:.3g} s{tail}"


# --------------------------------------------------------------------------- scene primitives
def scene(res: box.BoxResult, phase: float, yaw: float, pitch: float, show_sites: bool = True, full_paths: bool = True,
          view2d: bool = False) -> list[tuple]:
    """Primitives for the box view: ('circle', x, y, r, fill, line, w) / ('line', pts, color, w, dash) /
    ('rect', x1, y1, x2, y2, fill, line) / ('text', x, y, text, color, size, anchor, bold)."""
    half = res.side_cm / 2.0
    P: list[tuple] = []
    if view2d:                                                    # flat view straight onto the x-y plane (depth is dropped)
        yaw = pitch = 0.0
    kz = 1.2 if view2d else 1.0
    pr = lambda p: project(p[0], p[1], p[2], yaw, pitch, half, kz)
    corners = [project(*c, yaw, pitch, half, kz) for c in cube_corners(half)]
    if view2d:
        sq = [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]
        flat = []
        for x, y in sq:
            sx, sy, _ = project(x, y, 0.0, 0.0, 0.0, half, kz)
            flat += [sx, sy]
        P.append(("line", flat, EDGE_FRONT, 2, False))
        P.append(("text", 10, 44, "flat view: x across, y up (z collapsed)", "#9db4d4", 9, "nw", False))
    else:
        depth_cut = sorted(c[2] for c in corners)[3]                 # the nearer half of the corners is "front"
        for a, b in CUBE_EDGES:
            front = (corners[a][2] + corners[b][2]) / 2.0 > depth_cut - 1e-9
            P.append(("line", [corners[a][0], corners[a][1], corners[b][0], corners[b][1]],
                      EDGE_FRONT if front else EDGE_BACK, 2 if front else 1, False))
    if full_paths:                                                # everything each watched neutron has already done, faint
        for tr in res.traces:
            k = min(len(tr.steps), max(0, int(phase - tr.start_phase)))
            if phase < tr.start_phase or k < 1:
                continue
            pts = [(tr.x0, tr.y0, tr.z0)] + [st[:3] for st in tr.steps[:k]]
            flat = []
            for p in pts:
                sx, sy, _ = pr(p)
                flat += [sx, sy]
            P.append(("line", flat, "#1e3052", 1, False))
    if show_sites:                                                # nuclei that have really been struck
        for x, y, z, kind in struck_sites(res.traces, phase):
            sx, sy, _ = pr((x, y, z))
            P.append(("circle", sx, sy, 2.6, "", KIND_COLOR[kind], 1))
    states = [(i, neutron_state(t, phase), t) for i, t in enumerate(res.traces)]
    for i, s, tr in states:                                       # trails and persistent end markers first
        if s is None:
            continue
        if s["prev"] is not None:
            pts = [tr.steps[j - 1][:3] if j > 0 else (tr.x0, tr.y0, tr.z0) for j in range(s["trail_from"], int(phase - tr.start_phase) + 1)]
            pts.append(s["pos"])
            if len(pts) >= 2:
                flat = []
                for p in pts:
                    sx, sy, _ = pr(p)
                    flat += [sx, sy]
                P.append(("line", flat, "#3a5f9e", 1, False))
    for i, s, tr in states:
        if s is None:
            continue
        sx, sy, _ = pr(s["pos"])
        if s["done"] == "absorb":
            P.append(("line", [sx - 4, sy - 4, sx + 4, sy + 4], KIND_COLOR["absorb"], 2, False))
            P.append(("line", [sx - 4, sy + 4, sx + 4, sy - 4], KIND_COLOR["absorb"], 2, False))
            if s["age"] < FLASH:
                P.append(("circle", sx, sy, 6 + 14 * s["age"] / FLASH, "", KIND_COLOR["absorb"], 2))
        elif s["done"] == "leak":
            P.append(("circle", sx, sy, 3.2, KIND_COLOR["leak"], KIND_COLOR["leak"], 1))
        elif s["done"] in ("alive", "cap", "truncated"):
            P.append(("circle", sx, sy, 3.2, "", KIND_COLOR["alive"], 1))
        else:
            P.append(("circle", sx, sy, 3.6, energy_color(s["E"]), "#ffffff", 1))
            fl = s.get("flash")
            if fl and fl[1] < FLASH:
                kind, age, p0 = fl
                fx, fy, _ = pr(p0)
                P.append(("circle", fx, fy, 4 + 14 * age / FLASH, "", KIND_COLOR.get(kind, "#ffffff"), 2))
    # legend and text
    P.append(("text", 10, 10, f"{res.isotope}  -  neutrons of {box._fmt_e(res.energy_eV)}  -  cube {res.side_cm:g} cm, "
              f"{res.density_factor:.3g} x solid", "#ffffff", 12, "nw", True))
    if res.model_estimate:
        P.append(("text", 10, 28, "MODEL ESTIMATE (no evaluated data): paths follow physics-law cross sections", "#ffb347", 9, "nw", False))
    shown = len(res.traces)
    P.append(("text", 10, H - 56, f"Showing {shown} of {res.n_total} neutrons (statistics cover all). Rings = nuclei actually struck.", "#9db4d4", 9, "nw", False))
    P.append(("text", 10, H - 42, f"Time per segment on screen is schematic. Real flight so far: {path_length_cm(res.traces, phase):.4g} cm "
              f"(sum over watched neutrons).", "#9db4d4", 9, "nw", False))
    P.append(("text", 10, H - 24, caption(res.traces, phase), "#e6eefb", 10, "nw", False))
    lx = W - 150
    P.append(("text", lx, 10, "neutron energy", "#c9d6ea", 9, "nw", False))
    for k, (e, lab) in enumerate([(0.0253, "thermal"), (1e3, "keV"), (1e6, "MeV"), (1.4e7, "14 MeV")]):
        P.append(("circle", lx + 6 + 34 * k, 30, 4, energy_color(e), "#ffffff", 1))
        P.append(("text", lx + 6 + 34 * k, 38, lab, "#9db4d4", 7, "n", False))
    for k, (kind, lab) in enumerate([("elastic", "scatter"), ("react", "reaction"), ("absorb", "absorbed"), ("leak", "leaked")]):
        P.append(("circle", lx + 6, 58 + 13 * k, 3.5, KIND_COLOR[kind], KIND_COLOR[kind], 1))
        P.append(("text", lx + 16, 58 + 13 * k, lab, "#c9d6ea", 9, "w", False))
    return P


def tally_scene(res: box.BoxResult, phase: float, w: int = TW, h: int = H) -> list[tuple]:
    """Counts so far among the drawn neutrons, next to the final numbers for all of them."""
    t = tally_at(res.traces, phase)
    P: list[tuple] = [("text", 10, 10, "What has happened so far", "#ffffff", 13, "nw", True),
                      ("text", 10, 30, "among the neutrons you are watching", "#9db4d4", 9, "nw", False)]
    ends, kinds = t["ends"], t["kinds"]
    n_tr = max(1, len(res.traces))
    rows = [("absorbed", ends["absorb"], KIND_COLOR["absorb"]), ("leaked out", ends["leak"], KIND_COLOR["leak"]),
            ("still moving at the end time", ends["alive"] + ends["cap"], KIND_COLOR["alive"])]
    y = 56
    bw = w - 20
    for lab, k, col in rows:
        P.append(("text", 10, y, lab, "#e6eefb", 10, "nw", False))
        P.append(("text", w - 10, y, f"{k}/{len(res.traces)}", col, 9, "ne", False))
        P.append(("rect", 10, y + 16, 10 + bw, y + 24, "#1c2940", "#1c2940"))
        P.append(("rect", 10, y + 16, 10 + max(bw * k / n_tr, 1.0 if k else 0.0), y + 24, col, col))
        y += 36
    y += 6
    P.append(("text", 10, y, "collisions so far, by type", "#ffffff", 11, "nw", True))
    y += 20
    total = sum(t["mts"].values()) or 1
    lab_of = {None: "?"}
    for mt, c in sorted(t["mts"].items(), key=lambda kv: -kv[1])[:8]:
        name = "MT2 elastic scattering" if mt == 2 else f"MT{mt} {channels.name(mt)}"
        name = name if len(name) <= 34 else name[:33] + "\u2026"
        col = KIND_COLOR["elastic"] if mt == 2 else KIND_COLOR["absorb"] if mt in (102, 103, 107, 104, 105, 106) else KIND_COLOR["react"]
        P.append(("text", 10, y, name, "#e6eefb", 9, "nw", False))
        P.append(("text", w - 10, y, str(c), col, 9, "ne", False))
        P.append(("rect", 10, y + 13, 10 + bw, y + 19, "#1c2940", "#1c2940"))
        P.append(("rect", 10, y + 13, 10 + max(bw * c / total, 1.0), y + 19, col, col))
        y += 28
    f = scale_facts(res)
    lines = [f"All {res.n_total} neutrons, final: absorbed {res.absorbed}, leaked {res.leaked}, alive {res.alive_end}.",
             f"Heat {res.heat_MeV:.4g} MeV ({res.heat_per_source_MeV:.3g} per neutron).",
             f"True scales: {f['atoms_per_cm3']:.3g} atoms/cm3, {f['atoms_in_box']:.3g} atoms in the box, spacing {f['spacing_cm']:.2g} cm; "
             f"mean free path {f['mfp_cm']:.3g} cm = {f['mfp_in_spacings']:.3g} spacings; box = {f['side_in_mfp']:.3g} mean free paths; "
             f"spacing = {f['spacing_in_nuclear_diameters']:.3g} nuclear diameters."]
    wrapped = [ln for t_ in lines for ln in _wrap(t_, 42)]
    y = max(y, h - 14 * len(wrapped) - 8)
    for ln in wrapped:
        P.append(("text", 10, y, ln, "#9db4d4", 9, "nw", False))
        y += 14
    return P


# --------------------------------------------------------------------------- tkinter panel
if tk is not None:

    class _Painter:
        """Paints scene primitives; same format and behaviour as the single-atom visualizer's painter."""
        def __init__(self, family: str):
            self.family, self.fonts = family, {}

        def font(self, size: int, bold: bool):
            key = (size, bold)
            if key not in self.fonts:
                self.fonts[key] = tkfont.Font(family=self.family, size=size, weight="bold" if bold else "normal")
            return self.fonts[key]

        def paint(self, canvas, prims, k: float = 1.0):
            canvas.delete("all")
            for p in prims:
                t = p[0]
                if t == "circle":
                    _, x, y, r, fill, line, wd = p
                    canvas.create_oval((x - r) * k, (y - r) * k, (x + r) * k, (y + r) * k, fill=fill, outline=line,
                                       width=max(1, round(wd * k)))
                elif t == "line":
                    _, pts, color, wd, dash = p
                    canvas.create_line(*[v * k for v in pts], fill=color, width=max(1, round(wd * k)))
                elif t == "rect":
                    _, x1, y1, x2, y2, fill, line = p
                    canvas.create_rectangle(x1 * k, y1 * k, x2 * k, y2 * k, fill=fill, outline=line)
                elif t == "text":
                    _, x, y, text, color, size, anchor, bold = p
                    canvas.create_text(x * k, y * k, text=text, fill=color, font=self.font(max(4, round(size * k)), bold), anchor=anchor)

    class BoxVizPanel(ttk.Frame):
        """Play, pause, restart and rotate the recorded neutron histories of one box run."""

        def __init__(self, master):
            super().__init__(master, padding=6)
            self.res: box.BoxResult | None = None
            self.phase = 0.0
            self.yaw, self.pitch = 0.6, 0.45
            self.playing = True
            self._last = time.monotonic()
            self._drag = None
            self._alive = True
            self.k = 0.7
            self.painter = _Painter(tkfont.nametofont("TkDefaultFont").actual("family"))
            bar = ttk.Frame(self)
            bar.pack(side="bottom", fill="x", pady=(6, 0))
            self.play_btn = ttk.Button(bar, text="Pause", command=self.toggle)
            self.play_btn.pack(side="left")
            ttk.Button(bar, text="Restart", command=self.restart).pack(side="left", padx=6)
            ttk.Label(bar, text="Speed:").pack(side="left", padx=(10, 0))
            self.speed = tk.StringVar(value="Normal")
            ttk.Combobox(bar, textvariable=self.speed, values=list(SPEEDS), width=8, state="readonly").pack(side="left", padx=4)
            self.loop = tk.BooleanVar(value=True)
            ttk.Checkbutton(bar, text="Loop", variable=self.loop).pack(side="left", padx=8)
            self.spin = tk.BooleanVar(value=False)
            ttk.Checkbutton(bar, text="Slowly rotate", variable=self.spin).pack(side="left", padx=4)
            self.atoms = tk.BooleanVar(value=True)
            ttk.Checkbutton(bar, text="Struck nuclei", variable=self.atoms).pack(side="left", padx=4)
            ttk.Label(bar, text="View:").pack(side="left", padx=(10, 0))
            self.view = tk.StringVar(value="3D")
            vb = ttk.Combobox(bar, textvariable=self.view, values=["3D", "2D"], width=4, state="readonly")
            vb.pack(side="left", padx=4)
            vb.bind("<<ComboboxSelected>>", lambda e: self._paint())
            self.paths = tk.BooleanVar(value=True)
            ttk.Checkbutton(bar, text="Full paths", variable=self.paths).pack(side="left", padx=4)
            ttk.Label(bar, text="3D: drag to rotate").pack(side="right")
            # both canvases keep the scene's proportions: sized from the room `row` gets, centred, everything scaled by k
            row = ttk.Frame(self, width=round((W + TW + GAP) * 0.7), height=round(H * 0.7))
            row.pack(side="top", fill="both", expand=True)
            row.bind("<Configure>", self._on_resize)
            holder = ttk.Frame(row)
            holder.place(relx=0.5, rely=0.5, anchor="center")
            self.canvas = tk.Canvas(holder, width=round(W * self.k), height=round(H * self.k), bg=BG, highlightthickness=0)
            self.canvas.pack(side="left")
            self.tally = tk.Canvas(holder, width=round(TW * self.k), height=round(H * self.k), bg=BG, highlightthickness=0)
            self.tally.pack(side="left", padx=(GAP, 0))
            self.canvas.bind("<ButtonPress-1>", self._press)
            self.canvas.bind("<B1-Motion>", self._motion)
            self.bind("<Destroy>", self._on_destroy)
            self.after(40, self._tick)

        # --- data
        def set_result(self, res: box.BoxResult | None) -> None:
            self.res = res
            self.phase = 0.0
            self._last = time.monotonic()
            self._paint()

        # --- controls
        def toggle(self):
            self.playing = not self.playing
            self.play_btn.configure(text="Pause" if self.playing else "Play")
            self._last = time.monotonic()

        def restart(self):
            self.phase = 0.0
            self.playing = True
            self.play_btn.configure(text="Pause")
            self._last = time.monotonic()

        def _press(self, e):
            if self.view.get() == "2D":
                return
            self._drag = (e.x, e.y, self.yaw, self.pitch)

        def _motion(self, e):
            if self._drag is None:
                return
            x0, y0, yaw0, pitch0 = self._drag
            self.yaw = yaw0 + (e.x - x0) * 0.01
            self.pitch = max(-1.5, min(1.5, pitch0 + (e.y - y0) * 0.01))
            self._paint()

        def _on_resize(self, e):
            k = max(0.35, min(3.0, min((e.width - 2) / (W + TW + GAP), (e.height - 2) / H)))
            if abs(k - self.k) >= 0.01:
                self.k = k
                self.canvas.configure(width=round(W * k), height=round(H * k))
                self.tally.configure(width=round(TW * k), height=round(H * k))
                self._paint()

        def _on_destroy(self, e):
            if e.widget is self:
                self._alive = False

        # --- loop
        def _tick(self):
            if not self._alive:
                return
            now = time.monotonic()
            dt, self._last = now - self._last, now
            try:
                if self.res is not None and self.res.traces:
                    if self.spin.get():
                        self.yaw += 0.25 * dt
                    if self.playing:
                        self.phase += dt * SPEEDS.get(self.speed.get(), 5.0)
                        end = total_phase(self.res.traces) + FLASH + 4
                        if self.phase >= end:
                            if self.loop.get():
                                self.phase = 0.0
                            else:
                                self.phase = end
                                self.playing = False
                                self.play_btn.configure(text="Play")
                    self._paint()
                self.after(40, self._tick)
            except tk.TclError:
                self._alive = False

        def _paint(self):
            try:
                if self.res is None:
                    self.canvas.delete("all")
                    self.canvas.create_text(20, 20, anchor="nw", fill="#9db4d4", text="Run the box test to watch it here.")
                    self.tally.delete("all")
                    return
                if not self.res.traces:
                    self.canvas.delete("all")
                    self.canvas.create_text(20, 20, anchor="nw", fill="#9db4d4", text="No neutron paths were recorded for this run.")
                    return
                self.painter.paint(self.canvas, scene(self.res, self.phase, self.yaw, self.pitch, self.atoms.get(), self.paths.get(), self.view.get() == "2D"), self.k)
                self.painter.paint(self.tally, tally_scene(self.res, self.phase), self.k)
            except tk.TclError:
                self._alive = False


def main(isotope: str | None = None) -> int:
    """Open the box window on its Watch tab (python run_gui.py boxviz [ISOTOPE])."""
    from . import boxgui
    return boxgui.main(isotope, watch=True)
