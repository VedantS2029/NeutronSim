"""Step 2 of the isotope-explorer: periodic-table picker with +/- particle controls.

Layout of this file (same pattern as gui.py: logic first so it is testable without a display):
  1. data      - category colours, typical mass numbers (starting points only)
  2. ExplorerModel - all behaviour: pick an element, step protons/neutrons/electrons, parse typed input
  3. ExplorerApp   - the tkinter window; it only draws the model and forwards clicks

Physics estimates (step 3, physics.py) appear in status_lines(); measured-data comparison (step 4) plugs in there too.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import chem, compare, core, nphys, physics
from .scaling import SCREEN_FRACTION, TASKBAR_ALLOWANCE, autowrap, fit_scale, fit_window, screen_size
from .nuclide import MAX_Z, Nuclide, NuclideError, element_name, element_symbol, periodic_position

try:  # optional at import time, exactly as in gui.py
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover
    tk = None

# --------------------------------------------------------------------------- 1. data
# Starting mass number offered when you click an element. This is a convenience, NOT a claim about which isotope
# matters: it is the commonly cited most-abundant stable isotope, or for elements with no stable isotope the
# commonly cited long-lived one. Typed from reference memory, not read from a nuclide database - check a table
# (e.g. NUBASE/AME) before relying on any of these. Index = Z - 1.
TYPICAL_A = (
    1, 4, 7, 9, 11, 12, 14, 16, 19, 20,            # H-Ne
    23, 24, 27, 28, 31, 32, 35, 40,                 # Na-Ar
    39, 40, 45, 48, 51, 52, 55, 56, 59, 58, 63, 64, 69, 74, 75, 80, 79, 84,   # K-Kr
    85, 88, 89, 90, 93, 98, 99, 102, 103, 106, 107, 114, 115, 120, 121, 130, 127, 132,  # Rb-Xe
    133, 138, 139, 140, 141, 142, 145, 152, 153, 158, 159, 164, 165, 166, 169, 174, 175,  # Cs-Lu
    180, 181, 184, 187, 192, 193, 195, 197, 202, 205, 208, 209,                 # Hf-Bi
    209, 210, 222, 223, 226, 227, 232, 231, 238, 237, 244, 243, 247, 247, 251, 252, 257, 258, 259, 266,  # Po-Lr
    267, 268, 269, 270, 269, 278, 281, 282, 285, 286, 289, 290, 293, 294, 294,  # Rf-Og
)
assert len(TYPICAL_A) == MAX_Z

_CATEGORY_SETS = (
    ("noble gas", {2, 10, 18, 36, 54, 86, 118}),
    ("alkali metal", {3, 11, 19, 37, 55, 87}),
    ("alkaline earth", {4, 12, 20, 38, 56, 88}),
    ("halogen", {9, 17, 35, 53, 85, 117}),
    ("metalloid", {5, 14, 32, 33, 51, 52}),
    ("nonmetal", {1, 6, 7, 8, 15, 16, 34}),
    ("post-transition metal", {13, 31, 49, 50, 81, 82, 83, 84, 113, 114, 115, 116}),
)
CATEGORY_COLOURS = {
    "noble gas": "#c9b6e4", "alkali metal": "#f4a6a0", "alkaline earth": "#f6cf9a", "halogen": "#a9dcc4",
    "metalloid": "#bcd3a0", "nonmetal": "#9fd0ee", "post-transition metal": "#c8d3dd",
    "transition metal": "#f2e39b", "lanthanide": "#f3b8d6", "actinide": "#e8a0b8",
}


def category(z: int) -> str:
    for name, members in _CATEGORY_SETS:
        if z in members:
            return name
    if 57 <= z <= 71:
        return "lanthanide"
    if 89 <= z <= 103:
        return "actinide"
    return "transition metal"


def typical_isotope(z: int) -> Nuclide:
    element_symbol(z)                                  # validates z
    return Nuclide(z, TYPICAL_A[z - 1] - z)


def starting_isotope(z: int, keys=None) -> Nuclide:
    """What clicking element z selects: an isotope with real data if the element has one (that is what the green dot
    promises), otherwise the typical isotope. With several data isotopes the one nearest the typical A wins."""
    keys = data_isotopes() if keys is None else keys
    have = [Nuclide.parse(k) for k in keys if Nuclide.parse(k).Z == z]
    if not have:
        return typical_isotope(z)
    t = TYPICAL_A[z - 1]
    return min(have, key=lambda n: (abs(n.A - t), n.A))


def data_isotopes() -> frozenset[str]:
    """Isotope keys with real evaluated data on disk (read from the manifest; 'B-10', 'U-235', ...)."""
    try:
        return frozenset(json.loads((core.DATA_DIR / "MANIFEST.json").read_text())["isotopes"])
    except (OSError, ValueError, KeyError):
        return frozenset()


# --------------------------------------------------------------------------- 2. model
@dataclass
class ExplorerModel:
    nuclide: Nuclide = field(default_factory=lambda: typical_isotope(6))
    message: str = ""
    data_keys: frozenset = field(default_factory=data_isotopes)
    energy_eV: float = 0.0253            # incident neutron energy used for the measured-data block

    # ---- queries
    @property
    def has_data(self) -> bool:
        return self.nuclide.key in self.data_keys

    def elements_with_data(self) -> set[int]:
        return {Nuclide.parse(k).Z for k in self.data_keys}

    def status_lines(self) -> list[str]:
        n = self.nuclide
        lines = [n.describe()]
        if self.has_data:
            lines.append(f"  Real evaluated neutron data is available for {n.key} (ENDF/B-VIII.0 in this project).")
        else:
            lines.append(f"  No measured neutron data loaded for {n.key}. Only the model estimate below is available.")
        lines += physics.render(physics.estimate_nuclide(n))
        lines += chem.render(n)
        lines += compare.lines(n, self.energy_eV)
        if not self.has_data:
            lines += nphys.render(nphys.forecast(n, self.energy_eV))
        return lines

    def set_energy_label(self, label: str) -> bool:
        for name, e in compare.ENERGY_PRESETS:
            if name == label:
                self.energy_eV = e
                return True
        return False

    # ---- actions. Each returns True when something changed; failures set .message and change nothing.
    def _apply(self, new: Nuclide, what: str) -> bool:
        old, self.nuclide = self.nuclide, new
        notes = [what]
        if new.Z != old.Z:
            notes.append(f"element changed: {old.symbol} -> {new.symbol} ({new.element})")
        elif new.N != old.N:
            notes.append(f"isotope changed: {old.key} -> {new.key}")
        if new.charge != old.charge:
            notes.append(f"charge is now {'0' if new.charge == 0 else format(new.charge, '+d')}")
        self.message = "; ".join(notes)
        return True

    def select_element(self, z: int) -> bool:
        try:
            n = starting_isotope(z, self.data_keys)
            why = "an isotope with real data" if n.key in self.data_keys else "the typical isotope"
            return self._apply(n, f"selected {element_name(z)} (starting isotope A = {n.A}, {why})")
        except NuclideError as exc:
            self.message = str(exc)
            return False

    def step_protons(self, delta: int) -> bool:
        """Add/remove protons, leaving neutrons and electrons alone (so the charge changes)."""
        try:
            return self._apply(self.nuclide.with_protons(self.nuclide.Z + delta, keep_neutral=False),
                               f"protons {self.nuclide.Z} -> {self.nuclide.Z + delta}")
        except NuclideError as exc:
            self.message = str(exc)
            return False

    def step_neutrons(self, delta: int) -> bool:
        try:
            return self._apply(self.nuclide.with_neutrons(self.nuclide.N + delta),
                               f"neutrons {self.nuclide.N} -> {self.nuclide.N + delta}")
        except NuclideError as exc:
            self.message = str(exc)
            return False

    def step_electrons(self, delta: int) -> bool:
        try:
            return self._apply(self.nuclide.with_electrons(self.nuclide.electrons + delta),
                               f"electrons {self.nuclide.electrons} -> {self.nuclide.electrons + delta}")
        except NuclideError as exc:
            self.message = str(exc)
            return False

    def neutralise(self) -> bool:
        return self._apply(self.nuclide.neutralised(), "electrons set equal to protons")

    def set_from_text(self, text: str) -> bool:
        try:
            return self._apply(Nuclide.parse(text), f"entered {text.strip()!r}")
        except NuclideError as exc:
            self.message = str(exc)
            return False


# --------------------------------------------------------------------------- layout (pure, testable)
CELL, PAD = 42, 10
GRID_ROWS, GRID_COLS = 10, 18
CANVAS_W, CANVAS_H = GRID_COLS * CELL + 2 * PAD, GRID_ROWS * CELL + 2 * PAD
MIN_SCALE, MAX_SCALE = 0.2, 3.0          # how far the table may shrink / grow from its logical size


def cell_origin(z: int) -> tuple[int, int]:
    row, col = periodic_position(z)
    return PAD + (col - 1) * CELL, PAD + (row - 1) * CELL


def cell_at(x: float, y: float) -> int | None:
    """Element under a canvas click, or None for the gaps between/around cells."""
    col, row = int((x - PAD) // CELL) + 1, int((y - PAD) // CELL) + 1
    if x < PAD or y < PAD:
        return None
    for z in range(1, MAX_Z + 1):
        if periodic_position(z) == (row, col):
            return z
    return None


# --------------------------------------------------------------------------- 3. window
class _ExplorerUI:
    """Everything the window does; ExplorerApp (own window) and ExplorerWindow (inside the wizard) only differ in their base class."""

    def _init_ui(self, model: ExplorerModel | None, on_choose=None):
        self.model = model or ExplorerModel()
        self.on_choose = on_choose            # called with an isotope key like "U-235" when the user presses the wizard button
        self.title("neutronsim - isotope explorer")
        self.status_var, self.entry_var, self.msg_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.energy_var = tk.StringVar(value=compare.ENERGY_PRESETS[0][0])
        self.count_vars = {k: tk.StringVar() for k in ("protons", "neutrons", "electrons")}
        self.use_btn = None
        self._k = 1.0                           # canvas scale: drawn size / logical CANVAS_W x CANVAS_H size
        self._build()
        self._refresh()
        self._fit()

    def _build(self):
        # Left: the periodic table, scaled to fit its space. Right: the controls (counters, type-in box + Apply,
        # energy, wizard button) ABOVE the info text, which takes the remaining height and scrolls.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        size = screen_size(self)
        if size is not None:
            self._k = fit_scale(size[0] * SCREEN_FRACTION - 420, size[1] * SCREEN_FRACTION - TASKBAR_ALLOWANCE - 40,
                                CANVAS_W, CANVAS_H, min_k=MIN_SCALE, max_k=1.0)
        left = ttk.Frame(self)
        left.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.canvas = tk.Canvas(left, width=round(CANVAS_W * self._k), height=round(CANVAS_H * self._k),
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", self._on_resize)

        right = ttk.Frame(self)
        right.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")
        controls = ttk.Frame(right)
        controls.pack(side="top", fill="x")
        self._right = self._controls = controls
        stat = ttk.Frame(right)
        stat.pack(side="top", fill="both", expand=True, pady=(8, 0))
        try:
            bg = ttk.Style(self).lookup("TFrame", "background") or None
        except Exception:
            bg = None
        self.status_text = tk.Text(stat, wrap="word", width=34, height=6, borderwidth=0, highlightthickness=0,
                                   font="TkDefaultFont", state="disabled", **({"background": bg} if bg else {}))
        sbar = ttk.Scrollbar(stat, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=sbar.set)
        sbar.pack(side="right", fill="y")
        self.status_text.pack(side="left", fill="both", expand=True)
        helps = {"protons": "Protons\n(sets the ELEMENT)",
                 "neutrons": "Neutrons\n(sets the ISOTOPE)",
                 "electrons": "Electrons\n(sets the ION CHARGE)"}
        counters = ttk.Frame(controls)          # the three counters side by side to save height
        counters.pack(anchor="w")
        for col, key in enumerate(("protons", "neutrons", "electrons")):
            ttk.Label(counters, text=helps[key], justify="center").grid(row=0, column=col, padx=6)
            row = ttk.Frame(counters)
            row.grid(row=1, column=col, padx=6, pady=(0, 6))
            ttk.Button(row, text="\u2212", width=2, command=lambda k=key: self._bump(k, -1)).pack(side="left")
            ttk.Label(row, textvariable=self.count_vars[key], width=4, anchor="center").pack(side="left")
            ttk.Button(row, text="+", width=2, command=lambda k=key: self._bump(k, +1)).pack(side="left")
        ttk.Button(controls, text="Make neutral (electrons = protons)", command=self._neutralise).pack(anchor="w", pady=4)
        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(controls, text="Or type one (U-235, 235U, tritium, O-16 2-):").pack(anchor="w")
        erow = ttk.Frame(controls)          # own row: side="left" packing straight into `controls` would push every later widget sideways
        erow.pack(anchor="w")
        entry = ttk.Entry(erow, textvariable=self.entry_var, width=22)
        entry.pack(side="left")
        entry.bind("<Return>", lambda _e: self._apply_text())
        ttk.Button(erow, text="Apply", command=self._apply_text).pack(side="left", padx=4)
        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(controls, text="Neutron energy for the measured-data block:").pack(anchor="w")
        box = ttk.Combobox(controls, textvariable=self.energy_var, values=[n for n, _ in compare.ENERGY_PRESETS],
                           state="readonly", width=22)
        box.pack(anchor="w")
        self.energy_var.trace_add("write", lambda *_: self._on_energy())
        if self.on_choose:
            ttk.Separator(controls).pack(fill="x", pady=8)
            self.use_btn = ttk.Button(controls, text="Use this isotope in the wizard", command=self._use)
            self.use_btn.pack(anchor="w")
            ttk.Label(controls, text="(real data -> measured results; otherwise a labelled model estimate)",
                      wraplength=260, justify="left").pack(anchor="w")
        msg = ttk.Label(self, textvariable=self.msg_var, wraplength=700, justify="left")
        msg.grid(row=1, column=0, columnspan=2, padx=8, pady=(0, 8), sticky="ew")
        autowrap(msg, pad=16)

    def _fit(self):
        """Open at the natural size, but never bigger than the screen; stay freely resizable."""
        try:
            self.update_idletasks()
            fit_window(self, self.winfo_reqwidth(), self.winfo_reqheight(), min_w=200, min_h=150)
        except Exception:
            pass

    def _on_resize(self, e):
        k = max(MIN_SCALE, min(MAX_SCALE, min(e.width / CANVAS_W, e.height / CANVAS_H)))
        if abs(k - self._k) >= 0.01:
            self._k = k
            self._draw_grid()

    # ---- events
    def _on_click(self, event):
        z = cell_at(event.x / self._k, event.y / self._k)     # back to the table's logical coordinates
        if z is not None:
            self.model.select_element(z)
            self._refresh()

    def _bump(self, what: str, delta: int):
        getattr(self.model, f"step_{what}")(delta)
        self._refresh()

    def _use(self):
        self.on_choose(self.model.nuclide.key)

    def _on_energy(self):
        if self.model.set_energy_label(self.energy_var.get()):
            self._refresh()

    def _neutralise(self):
        self.model.neutralise()
        self._refresh()

    def _apply_text(self):
        if self.model.set_from_text(self.entry_var.get()):
            self.entry_var.set("")
        self._refresh()

    # ---- drawing
    def _refresh(self):
        m, n = self.model, self.model.nuclide
        self.status_var.set("\n".join(m.status_lines()))
        self._show_status(self.status_var.get())
        self.msg_var.set(m.message)
        if self.use_btn is not None:
            self.use_btn.state(["!disabled"])
        for key, val in (("protons", n.Z), ("neutrons", n.N), ("electrons", n.electrons)):
            self.count_vars[key].set(str(val))
        self._draw_grid()

    def _show_status(self, text: str):
        t = getattr(self, "status_text", None)
        if t is not None:
            t.configure(state="normal")
            t.delete("1.0", "end")
            t.insert("1.0", text)
            t.configure(state="disabled")

    def _draw_grid(self):
        """Paint the periodic table at logical size, then scale it to the canvas (fonts and line widths by hand)."""
        m, n, k = self.model, self.model.nuclide, self._k
        c = self.canvas
        c.delete("all")
        fs = lambda pt: max(3, round(pt * k))          # noqa: E731 - font size at the current scale
        with_data = m.elements_with_data()
        for z in range(1, MAX_Z + 1):
            x, y = cell_origin(z)
            sel = z == n.Z
            c.create_rectangle(x + 1, y + 1, x + CELL - 1, y + CELL - 1, fill=CATEGORY_COLOURS[category(z)],
                               outline="#222222" if sel else "#888888", width=max(1, round((3 if sel else 1) * k)))
            c.create_text(x + CELL / 2, y + CELL / 2 - 4, text=element_symbol(z), fill="#111111",
                          font=("Helvetica", fs(12), "bold" if sel else "normal"))
            c.create_text(x + 4, y + 8, text=str(z), fill="#444444", anchor="w", font=("Helvetica", fs(7)))
            if z in with_data:
                c.create_oval(x + CELL - 10, y + 4, x + CELL - 4, y + 10, fill="#1a7f37", outline="")
        lx, ly = cell_origin(57)[0], cell_origin(57)[1] - CELL // 2
        c.create_text(PAD, PAD + 7.5 * CELL, text="f-block (La-Yb, Ac-No)", anchor="w", fill="#666666",
                      font=("Helvetica", fs(8)))
        c.create_text(CANVAS_W - PAD, CANVAS_H - 4, text="\u25cf = real neutron data available for an isotope of this element",
                      anchor="e", fill="#666666", font=("Helvetica", fs(8)))
        c.scale("all", 0, 0, k, k)


class ExplorerApp(_ExplorerUI, tk.Tk if tk else object):  # type: ignore[misc]
    def __init__(self, model: ExplorerModel | None = None):
        super().__init__()
        self._init_ui(model)


class ExplorerWindow(_ExplorerUI, tk.Toplevel if tk else object):  # type: ignore[misc]
    """The same explorer as a child window of the wizard; pressing the wizard button hands an isotope key back and closes it."""
    def __init__(self, master, model: ExplorerModel | None = None, on_choose=None):
        super().__init__(master)
        self._init_ui(model, on_choose)

    def _use(self):
        key = self.model.nuclide.key
        self.destroy()
        self.on_choose(key)


def main(initial: str | None = None) -> int:
    import sys
    model = ExplorerModel()
    if initial and not model.set_from_text(initial):          # report bad input even where there is no display
        print(f"error: {model.message}", file=sys.stderr)
        return 2
    if tk is None:
        print("error: tkinter is not available in this Python (install python3-tk).", file=sys.stderr)
        return 2
    try:
        app = ExplorerApp(model)
    except tk.TclError as exc:
        print(f"error: cannot open a window ({exc}).", file=sys.stderr)
        return 2
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
