"""Step B2: the box test as a window - density slider, live statistics, population and heat plots.

Same layout as the other windows: BoxModel holds all behaviour (testable without a display); BoxWindow / BoxApp
only draw it. tkinter is optional at import time, exactly as in gui.py. The engine is box.py (read its header for
scope and assumptions: any isotope except fissile / actinide ones - isotopes without evaluated data run on the labelled
physics-law model estimate; heat is local; fission is out of scope by design).

Two tabs: "Statistics" (numbers and plots) and "Watch" (the recorded neutron paths animated inside the box, boxviz.py).

Open it:  python run_gui.py boxgui [ISOTOPE]   |   python run_gui.py boxviz [ISOTOPE]   (opens on the Watch tab)
          |   wizard: Tools > Box test...   or the "Box test..." button on the wizard's Results page.
"""
from __future__ import annotations

import math
import queue
import threading

from . import box, core
from .scaling import autowrap, fit_window

try:  # optional at import time
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover
    tk = None

SLIDER_MIN = -4.0                                    # log10 of density relative to the solid
SLIDER_MAX = math.log10(box.MAX_DENSITY_FACTOR)
SLIDER_DEFAULT = -2.0
LIVE_N = 500                                         # default neutrons: quick enough to re-run while the slider moves
COMMON_MODEL_ISOTOPES = ["Li-6", "C-12", "O-16", "Na-23", "Fe-56", "Cu-63", "Cd-113", "Gd-157", "W-184", "Pb-208"]
DEFAULT_TRACE = 60                                   # neutron histories recorded for the Watch tab
PRESETS = [("Thermal", "0.0253", "eV"), ("1 keV", "1", "keV"), ("1 MeV", "1", "MeV"), ("14 MeV", "14", "MeV")]
COLORS = {"alive": "#3b7dd8", "absorbed": "#d6453d", "leaked": "#2e9e5b", "heat": "#c98a1b"}
FOOT_NOTE = ("Educational model. Any isotope except fissile / actinide ones (fission is out of scope by design); isotopes without "
             "evaluated data use the labelled physics-law model estimate. Gammas and charged particles are assumed to stop inside the "
             "box, so real heating of a small block would be lower.")
MODEL_NOTE = "MODEL ESTIMATE - no evaluated data for this isotope: cross sections come from physics laws and can be off by large factors."


def _boxviz():
    """Imported on demand (looked up at call time), like gui._viz()."""
    from . import boxviz
    return boxviz


# --------------------------------------------------------------------------- model (no GUI)
class BoxModel:
    def __init__(self, isotope: str = "B-10", energy_eV: float | None = None):
        ok = box.boxable_isotopes()
        self.isotopes = ok + [i for i in COMMON_MODEL_ISOTOPES if i not in ok]
        try:
            isotope = core.normalize_isotope(isotope)
        except core.InputError:
            pass
        why = box.refusal_reason(isotope)
        self.isotope = isotope if not why else ok[0]
        self.initial_note = why or ""
        self.energy_text, self.unit = "0.0253", "eV"
        if energy_eV:
            self.energy_text, self.unit = f"{energy_eV:g}", "eV"
        self.slider = SLIDER_DEFAULT
        self.side_text, self.n_text, self.source = "10", str(LIVE_N), "center"
        self.trace_text = str(DEFAULT_TRACE)
        self.result: box.BoxResult | None = None

    def set_isotope(self, text: str) -> None:
        """Accept what was typed ('fe56', 'Fe-56', '56Fe'); keep the raw text if it cannot be parsed (check() then says why)."""
        try:
            self.isotope = core.normalize_isotope(text)
        except core.InputError:
            self.isotope = text.strip()

    def trace_count(self) -> int:
        try:
            k = int(self.trace_text)
        except ValueError:
            return DEFAULT_TRACE
        return max(0, min(k, box.MAX_TRACES))

    # density
    @property
    def density_factor(self) -> float:
        return 10.0 ** self.slider

    def set_slider(self, v: float) -> None:
        self.slider = min(SLIDER_MAX, max(SLIDER_MIN, float(v)))

    def density_text(self) -> str:
        g = self.density_factor * box.reference_density(self.isotope)
        n = box.atoms_per_cm3(g, box.isotope_awr(self.isotope))
        return f"{self.density_factor:.3g} x solid  =  {g:.3g} g/cm3  =  {n:.2e} atoms/cm3"

    def set_preset(self, text: str, unit: str) -> None:
        self.energy_text, self.unit = text, unit

    def check(self) -> str | None:
        why = box.refusal_reason(self.isotope)
        if why:
            return why
        try:
            int(self.trace_text)
        except ValueError:
            return f"number of neutrons to watch {self.trace_text!r} is not a whole number"
        try:
            core.to_eV(float(self.energy_text), self.unit)
        except ValueError:
            return f"energy {self.energy_text!r} is not a number"
        except core.InputError as exc:
            return str(exc)
        try:
            side = float(self.side_text)
        except ValueError:
            return f"box side {self.side_text!r} is not a number"
        if not (box.MIN_SIDE_CM <= side <= box.MAX_SIDE_CM):
            return f"box side must be between {box.MIN_SIDE_CM} and {box.MAX_SIDE_CM} cm"
        try:
            n = int(self.n_text)
        except ValueError:
            return f"number of neutrons {self.n_text!r} is not a whole number"
        if not (1 <= n <= box.MAX_NEUTRONS):
            return f"number of neutrons must be between 1 and {box.MAX_NEUTRONS}"
        return None

    def run(self) -> box.BoxResult:
        err = self.check()
        if err:
            raise core.UnsupportedError(err)
        return box.run_box(self.isotope, core.to_eV(float(self.energy_text), self.unit), density_factor=self.density_factor,
                           side_cm=float(self.side_text), n=int(self.n_text), source=self.source,
                           trace=min(self.trace_count(), int(self.n_text)))


def stat_rows(r: box.BoxResult) -> list[tuple[str, str]]:
    pct = lambda k: f"{k}  ({100.0 * k / r.n_total:.1f} %)"
    born = f"  (+{r.n_born} born by (n,2n)-type reactions)" if r.n_born else ""
    return [
        ("Neutrons released", f"{r.n_source}{born}"),
        ("Left in the box (alive)", pct(r.alive_end) + (f"  at {r.t_end_s:g} s" if r.alive_end else "")),
        ("Absorbed", pct(r.absorbed)),
        ("Leaked out", pct(r.leaked)),
        ("Collisions", f"{r.collisions}  ({r.collisions / r.n_source:.3g} per neutron)"),
        ("Heat deposited", f"{r.heat_MeV:.6g} MeV  =  {r.heat_J:.4g} J"),
        ("Heat per released neutron", f"{r.heat_per_source_MeV:.4g} MeV"),
        ("Mean free path at start", f"{r.mean_free_path_cm:.4g} cm  (box side {r.side_cm:g} cm)"),
        ("Box contents", f"{r.box_mass_g:.4g} g,  {r.atoms_per_cm3:.3e} atoms/cm3"),
        ("Energy check (should be ~0)", f"{r.closure_MeV:.2g} MeV"),
    ]


def plot_series(r: box.BoxResult) -> dict:
    """Time axis and curves trimmed to where something still changes (the log-time tail would be a flat line)."""
    def last_change(vals):
        k = 0
        for i in range(1, len(vals)):
            if vals[i] != vals[i - 1]:
                k = i
        return k
    cut = max(last_change(r.alive_t), last_change(r.absorbed_t), last_change(r.leaked_t), last_change(r.heat_MeV_t))
    cut = min(len(r.times_s), cut + 3)
    tot = r.n_total
    return {"t": r.times_s[:cut],
            "alive": [100.0 * v / tot for v in r.alive_t[:cut]],
            "absorbed": [100.0 * v / tot for v in r.absorbed_t[:cut]],
            "leaked": [100.0 * v / tot for v in r.leaked_t[:cut]],
            "heat": r.heat_MeV_t[:cut]}


# --------------------------------------------------------------------------- window
def _draw_plot(cv, t, curves, title, ylabel, ymax=None):
    """Log-time line plot on a tk Canvas. curves = [(values, colour, label)]."""
    try:
        w = max(int(cv.winfo_width()), 320)
        h = max(int(cv.winfo_height()), 180)
    except Exception:
        w, h = 400, 220
    cv.delete("all")
    left, right, top, bottom = 52, 12, 24, 34
    cv.create_text(w // 2, 10, text=title, font=("TkDefaultFont", 10, "bold"))
    if len(t) < 2:
        cv.create_text(w // 2, h // 2, text="no events")
        return
    x0, x1 = math.log10(t[0]), math.log10(t[-1])
    x1 = x1 if x1 > x0 else x0 + 1
    top_v = ymax if ymax else max(1e-12, max(max(v) for v, _, _ in curves)) * 1.05
    px = lambda tt: left + (math.log10(tt) - x0) / (x1 - x0) * (w - left - right)
    py = lambda y: h - bottom - (y / top_v) * (h - top - bottom)
    cv.create_line(left, top, left, h - bottom, fill="#888")
    cv.create_line(left, h - bottom, w - right, h - bottom, fill="#888")
    for k in range(math.ceil(x0), math.floor(x1) + 1):
        xx = left + (k - x0) / (x1 - x0) * (w - left - right)
        cv.create_line(xx, h - bottom, xx, h - bottom + 4, fill="#888")
        cv.create_text(xx, h - bottom + 14, text=f"1e{k}", font=("TkDefaultFont", 8))
    for frac in (0, 0.5, 1.0):
        yy = py(top_v * frac)
        cv.create_line(left - 4, yy, left, yy, fill="#888")
        cv.create_text(left - 6, yy, text=f"{top_v * frac:.3g}", anchor="e", font=("TkDefaultFont", 8))
    cv.create_text(w // 2, h - 8, text="time (s)", font=("TkDefaultFont", 8))
    cv.create_text(6, top - 10, text=ylabel, anchor="w", font=("TkDefaultFont", 8))
    for i, (vals, colour, label) in enumerate(curves):
        pts = []
        for tt, v in zip(t, vals):
            pts += [px(tt), py(v)]
        if len(pts) >= 4:
            cv.create_line(*pts, fill=colour, width=2)
        cv.create_text(w - right - 4, top + 10 + 12 * i, text=label, fill=colour, anchor="e", font=("TkDefaultFont", 9))


class _BoxUI:
    """Everything the standalone app and the child window share (mixed into a Tk / Toplevel)."""

    def _init_ui(self, model: BoxModel):
        self.model = model
        self._q: queue.Queue = queue.Queue()
        self._gen = 0
        self._pending = None
        self._busy = False
        self.title("Box test - neutrons inside a block of one isotope")
        self.vars: dict = {}
        pad = {"padx": 8, "pady": 3}

        row1 = ttk.Frame(self)
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Isotope").pack(side="left")
        self.iso_var = tk.StringVar(self, model.isotope)
        iso = ttk.Combobox(row1, textvariable=self.iso_var, values=model.isotopes, width=9)      # editable: type any isotope
        iso.pack(side="left", padx=(4, 12))
        iso.bind("<<ComboboxSelected>>", lambda _e: self._changed())
        iso.bind("<Return>", lambda _e: self._changed())
        ttk.Label(row1, text="Neutron energy").pack(side="left")
        self.e_var = tk.StringVar(self, model.energy_text)
        ttk.Entry(row1, textvariable=self.e_var, width=9).pack(side="left", padx=4)
        self.u_var = tk.StringVar(self, model.unit)
        ttk.Combobox(row1, textvariable=self.u_var, values=list(core.UNITS), width=5, state="readonly").pack(side="left")
        for label, val, unit in PRESETS:
            ttk.Button(row1, text=label, width=8, command=lambda v=val, u=unit: self._preset(v, u)).pack(side="left", padx=2)

        row2 = ttk.Frame(self)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Density").pack(side="left")
        self.scale = ttk.Scale(row2, from_=SLIDER_MIN, to=SLIDER_MAX, value=model.slider, command=self._slider_moved)
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.dens_lbl = ttk.Label(row2, text="", width=58)
        self.dens_lbl.pack(side="left")

        row3 = ttk.Frame(self)
        row3.pack(fill="x", **pad)
        ttk.Label(row3, text="Box side (cm)").pack(side="left")
        self.s_var = tk.StringVar(self, model.side_text)
        ttk.Entry(row3, textvariable=self.s_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="Neutrons").pack(side="left")
        self.n_var = tk.StringVar(self, model.n_text)
        ttk.Entry(row3, textvariable=self.n_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="Released from").pack(side="left")
        self.src_var = tk.StringVar(self, model.source)
        ttk.Combobox(row3, textvariable=self.src_var, values=["center", "uniform"], width=8, state="readonly").pack(side="left", padx=4)
        ttk.Label(row3, text="Watch").pack(side="left")
        self.t_var = tk.StringVar(self, model.trace_text)
        ttk.Entry(row3, textvariable=self.t_var, width=5).pack(side="left", padx=(4, 12))
        self.live_var = tk.BooleanVar(self, True)
        ttk.Checkbutton(row3, text="Re-run while the slider moves", variable=self.live_var).pack(side="left", padx=12)
        self.run_btn = ttk.Button(row3, text="Run", command=self._run)
        self.run_btn.pack(side="right")

        self.msg = ttk.Label(self, text=model.initial_note, foreground="#d6453d")
        self.msg.pack(fill="x", **pad)
        autowrap(self.msg)
        self.model_note = ttk.Label(self, text="", foreground="#c98a1b")
        self.model_note.pack(fill="x", padx=8)
        autowrap(self.model_note)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, **pad)
        tab_stats, tab_watch = ttk.Frame(self.tabs), ttk.Frame(self.tabs)
        self.tabs.add(tab_stats, text="Statistics")
        self.tabs.add(tab_watch, text="Watch the neutrons")
        self.tab_watch = tab_watch
        self.viz_panel = _boxviz().BoxVizPanel(tab_watch)
        self.viz_panel.pack(fill="both", expand=True)

        mid = ttk.Frame(tab_stats)
        mid.pack(fill="both", expand=True, **pad)
        stats = ttk.Frame(mid)
        stats.pack(side="left", fill="y", padx=(0, 10))
        for i, (name, _v) in enumerate(stat_rows(_blank_result())):
            ttk.Label(stats, text=name).grid(row=i, column=0, sticky="w", pady=1)
            sv = tk.StringVar(self, "-")
            self.vars[name] = sv
            ttk.Label(stats, textvariable=sv).grid(row=i, column=1, sticky="w", padx=(10, 0))
        plots = ttk.Frame(mid)
        plots.pack(side="left", fill="both", expand=True)
        self.cv_pop = tk.Canvas(plots, width=420, height=220, background="white", highlightthickness=1)
        self.cv_pop.pack(fill="both", expand=True)
        self.cv_heat = tk.Canvas(plots, width=420, height=220, background="white", highlightthickness=1)
        self.cv_heat.pack(fill="both", expand=True, pady=(6, 0))
        for cv in (self.cv_pop, self.cv_heat):
            cv.bind("<Configure>", lambda _e: self._redraw())

        self.detail = tk.Text(tab_stats, wrap="none", height=9, width=60)
        self.detail.pack(fill="both", expand=False, **pad)
        note = ttk.Label(self, text=FOOT_NOTE, foreground="#8a8a8a")
        note.pack(fill="x", **pad)
        autowrap(note)
        fit_window(self, 980, 880, min_w=640, min_h=520)
        self._sync_density_label()
        self.after(100, self._poll)
        if getattr(self, "_start_on_watch", False):
            self.tabs.select(tab_watch)
        if not model.initial_note:
            self.after(150, self._run)

    # ---- reading the widgets into the model
    def _pull(self):
        m = self.model
        m.set_isotope(self.iso_var.get())
        m.trace_text = self.t_var.get()
        m.energy_text, m.unit = self.e_var.get(), self.u_var.get()
        m.side_text, m.n_text, m.source = self.s_var.get(), self.n_var.get(), self.src_var.get()

    def _sync_density_label(self):
        try:
            self.dens_lbl.configure(text=self.model.density_text())
        except Exception as exc:
            self.dens_lbl.configure(text=str(exc))

    def _preset(self, val, unit):
        self.e_var.set(val)
        self.u_var.set(unit)
        self._changed()

    def _changed(self):
        self._pull()
        self.msg.configure(text="")
        self._sync_density_label()
        self._run()

    def _slider_moved(self, value):
        self.model.set_slider(float(value))
        self._sync_density_label()
        if self.live_var.get():
            if self._pending is not None:
                try:
                    self.after_cancel(self._pending)
                except Exception:
                    pass
            self._pending = self.after(250, self._run)       # wait until the slider rests, then run once

    # ---- running (the engine works in a thread; results are delivered on the UI thread by _poll)
    def _run(self):
        self._pending = None
        self._pull()
        err = self.model.check()
        if err:
            self.msg.configure(text=err)
            return
        self.msg.configure(text="")
        self._gen += 1
        gen = self._gen
        snapshot = BoxModel(self.model.isotope)
        snapshot.__dict__.update(self.model.__dict__)

        def work():
            try:
                self._q.put((gen, snapshot.run()))
            except Exception as exc:                          # shown in the window, never swallowed
                self._q.put((gen, exc))
        self._busy = True
        self.run_btn.state(["disabled"])
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                gen, out = self._q.get_nowait()
                if gen != self._gen:
                    continue                                  # an older run: the slider has moved on
                self._busy = False
                self.run_btn.state(["!disabled"])
                self._show(out)
        except queue.Empty:
            pass
        try:
            self.after(100, self._poll)
        except Exception:
            pass

    def _show(self, out):
        if isinstance(out, (core.InputError, core.UnsupportedError)):
            self.msg.configure(text=str(out))
            return
        if isinstance(out, Exception):
            self.msg.configure(text=f"unexpected error: {out!r}")
            return
        self.model.result = out
        self.model_note.configure(text=MODEL_NOTE if out.model_estimate else "")
        self.viz_panel.set_result(out)
        for name, val in stat_rows(out):
            self.vars[name].set(val)
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", box.render(out))
        self.detail.configure(state="disabled")
        self._redraw()

    def _redraw(self):
        r = self.model.result
        if r is None:
            return
        s = plot_series(r)
        _draw_plot(self.cv_pop, s["t"], [(s["alive"], COLORS["alive"], "alive"), (s["absorbed"], COLORS["absorbed"], "absorbed"),
                                         (s["leaked"], COLORS["leaked"], "leaked")],
                   "Where the neutrons are", "% of all neutrons", ymax=100.0)
        _draw_plot(self.cv_heat, s["t"], [(s["heat"], COLORS["heat"], "heat in the box")],
                   "Heat generated in the box (cumulative)", "MeV")


def _blank_result() -> box.BoxResult:
    return box.BoxResult("", 0.0, "center", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, 0, 0, 0, 0, 0.0, 0, {}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


if tk is not None:
    class BoxApp(_BoxUI, tk.Tk):
        def __init__(self, model: BoxModel | None = None, watch: bool = False):
            super().__init__()
            self._start_on_watch = watch
            self._init_ui(model or BoxModel())

    class BoxWindow(_BoxUI, tk.Toplevel):
        """The same window as a child of the wizard."""
        def __init__(self, master, model: BoxModel | None = None, watch: bool = False):
            super().__init__(master)
            self._start_on_watch = watch
            self._init_ui(model or BoxModel())
else:  # pragma: no cover
    BoxApp = BoxWindow = None


def open_window(parent, isotope: str | None = None, energy_eV: float | None = None):
    return BoxWindow(parent, BoxModel(isotope or "B-10", energy_eV))


def main(initial: str | None = None, watch: bool = False) -> int:
    import sys
    iso = core.normalize_isotope(initial) if initial else "B-10"
    why = box.refusal_reason(iso)
    if why:
        print(f"error: {why}", file=sys.stderr)
        return 2
    if tk is None:
        print("error: tkinter is not available in this Python (install python3-tk).", file=sys.stderr)
        return 2
    try:
        app = BoxApp(BoxModel(iso), watch=watch)
    except tk.TclError as exc:
        print(f"error: cannot open a window ({exc}).", file=sys.stderr)
        return 2
    app.mainloop()
    return 0
