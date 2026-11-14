"""Step-by-step GUI setup wizard for neutronsim (tkinter, standard library only).

The wizard collects: acknowledgement -> isotope -> energy -> what to compute -> review -> results.
All logic lives in WizardModel (no GUI needed, unit-testable); the widgets below only display it.
Output text is produced by the same renderers the command line uses, so numbers, provenance
tags and disclaimers are identical in both front ends.
"""
from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from . import core, estsim, nphys, notes
from .nuclide import Nuclide, NuclideError
from .cli import BANNER, render_sample, render_xs
from .scaling import autowrap, fit_window

try:  # tkinter is optional at import time so WizardModel works on headless installs
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter import font as tkfont
except ImportError:  # pragma: no cover - depends on the Python build
    tk = None

PRESET_ENERGIES = [("Thermal (0.0253 eV)", "0.0253", "eV"),
                   ("1 keV", "1", "keV"),
                   ("1 MeV", "1", "MeV"),
                   ("14 MeV", "14", "MeV")]
DEFAULT_SAMPLES = 10_000
ESTIMATE_E_MIN_eV, ESTIMATE_E_MAX_eV = 1e-5, 2e7   # where the physics-law forecast (nphys) may be asked, for isotopes without data


def _viz():
    """Imported on demand (and looked up at call time) so the wizard works without the visualizer."""
    from . import viz
    return viz
def _explorer():
    """Imported on demand (and looked up at call time), like _viz()."""
    from . import explorer
    return explorer
def _boxgui():
    """Imported on demand, like _viz() and _explorer()."""
    from . import boxgui
    return boxgui
MUTED = "#8a8a8a"   # mid-grey / mid-red: readable in both light and dark appearance on macOS
ERROR = "#d6453d"


# --------------------------------------------------------------------------- model (no GUI)
@dataclass
class IsotopeInfo:
    name: str
    library: str
    e_min_eV: float
    e_max_eV: float
    mts: list[int]


@dataclass
class RunOutput:
    text: str
    supported: bool
    json_text: str
    result: core.Result | None = None


_DATASETS: dict[str, core.Dataset] = {}


def load_dataset(isotope: str) -> core.Dataset:
    """Load (and cache) one isotope. Verifies the data file hash on first load."""
    if isotope not in _DATASETS:
        _DATASETS[isotope] = core.Dataset(isotope)
    return _DATASETS[isotope]


def supported_isotopes() -> list[str]:
    man = json.loads((core.DATA_DIR / "MANIFEST.json").read_text())
    return sorted(man["isotopes"])


def isotope_info(isotope: str) -> IsotopeInfo:
    ds = load_dataset(isotope)
    lo, hi = ds.energy_range_eV
    return IsotopeInfo(isotope, ds.library, lo, hi, [2] + ds.partial_mts)


@dataclass
class WizardModel:
    acknowledged: bool = False
    isotope: str = ""
    energy_text: str = "0.0253"
    unit: str = "eV"
    mode: str = "xs"                 # "xs" or "sample"
    n_text: str = str(DEFAULT_SAMPLES)
    seed_text: str = ""

    # ---- per-step validation; each returns None when fine, else a message for the user
    def check_acknowledged(self) -> str | None:
        return None if self.acknowledged else "Please confirm you have read the notice to continue."

    @property
    def has_data(self) -> bool:
        """True when real evaluated data exist for the chosen isotope; otherwise only a model estimate can be run."""
        return self.isotope in supported_isotopes()

    def nuclide(self) -> Nuclide:
        return Nuclide.parse(self.isotope)

    def check_isotope(self) -> str | None:
        if not self.isotope:
            return "Choose a target isotope."
        if self.has_data:
            return None
        try:
            self.nuclide()
        except NuclideError as exc:
            return f"{self.isotope}: {exc}"
        return None          # a valid nuclide with no evaluated data: runs as a labelled MODEL ESTIMATE, never as data

    def energy_eV(self) -> float:
        return core.to_eV(self.energy_text.strip(), self.unit)

    def check_energy(self) -> str | None:
        try:
            e = self.energy_eV()
        except core.InputError as exc:
            return str(exc)
        if self.isotope and not self.has_data:
            if not (ESTIMATE_E_MIN_eV <= e <= ESTIMATE_E_MAX_eV):
                return (f"{e:g} eV is outside the model-estimate range {ESTIMATE_E_MIN_eV:g} eV to "
                        f"{ESTIMATE_E_MAX_eV:g} eV (above 1 keV it is an optical-model + Hauser-Feshbach estimate, checked against data only "
                        "for four isotopes up to 14 MeV).")
        elif self.isotope:
            info = isotope_info(self.isotope)
            if not (info.e_min_eV <= e <= info.e_max_eV):
                return (f"{e:g} eV is outside the evaluated range for {self.isotope}: "
                        f"{info.e_min_eV:g} to {info.e_max_eV:g} eV. No extrapolation is performed.")
        return None

    def sampling(self) -> tuple[int, int | None]:
        try:
            n = int(self.n_text.strip())
        except ValueError:
            raise core.InputError("number of samples must be a whole number") from None
        if not (1 <= n <= core.MAX_SAMPLES):
            raise core.InputError(f"number of samples must be between 1 and {core.MAX_SAMPLES:,}")
        seed_s = self.seed_text.strip()
        try:
            seed = int(seed_s) if seed_s else None
        except ValueError:
            raise core.InputError("seed must be a whole number (or leave it blank)") from None
        return n, seed

    def check_mode(self) -> str | None:
        if self.mode not in ("xs", "sample"):
            return "Choose what to compute."
        if self.mode == "sample" and self.isotope and not self.has_data:
            try:
                why = estsim.can_simulate(self.nuclide(), self.energy_eV())
            except (NuclideError, core.InputError):
                why = None        # bad isotope/energy are reported by their own steps
            if why:
                return why
        if self.mode == "sample":
            try:
                self.sampling()
            except core.InputError as exc:
                return str(exc)
        return None

    def summary(self) -> str:
        lines = [f"Target isotope:   {self.isotope}" + ("" if self.has_data else "   (NO evaluated data: model estimate only)"),
                 f"Neutron energy:   {self.energy_text.strip()} {self.unit}  (= {self.energy_eV():.6g} eV)"]
        if self.mode == "xs":
            lines.append("Compute:          cross sections and interaction probabilities")
        else:
            n, seed = self.sampling()
            lines.append(f"Compute:          cross sections + {n:,} random draws "
                         f"(seed {seed if seed is not None else 'random'})")
        return "\n".join(lines)

    def run_estimate(self) -> RunOutput:
        """Isotope without evaluated data: the physics-law forecast of nphys, headed so it cannot be mistaken for data."""
        if self.mode == "sample":
            n, seed = self.sampling()
            fc, res, sample = estsim.simulate(self.nuclide(), self.energy_eV(), n, seed)
            payload = {"model_estimate": asdict(fc), "sample": sample}
            return RunOutput(estsim.render_simulation(fc, res, sample), False, json.dumps(payload, indent=2, default=str), res)
        fc = nphys.forecast(self.nuclide(), self.energy_eV())
        head = [f"NO EVALUATED DATA for {self.isotope} in this project - this is a MODEL ESTIMATE, not an evaluated or measured result.",
                "Nothing below carries a dataset, version or uncertainty from a nuclear data library. Educational model only.", ""]
        text = "\n".join(head + nphys.render(fc))
        try:
            why = estsim.can_simulate(self.nuclide(), self.energy_eV())
        except Exception:
            why = None
        if why:
            text += f"\n\nVisualizer: the nucleus can be shown, but no neutron can be fired here. {why}"
        return RunOutput(text, False, json.dumps({"model_estimate": asdict(fc)}, indent=2, default=str),
                         estsim.visual_result(self.nuclide(), self.energy_eV()))

    # ---- the actual calculation (same code path as the CLI)
    def run(self) -> RunOutput:
        for check in (self.check_acknowledged, self.check_isotope, self.check_energy, self.check_mode):
            msg = check()
            if msg:
                raise core.InputError(msg)
        if not self.has_data:
            return self.run_estimate()
        ds = load_dataset(self.isotope)
        res = core.evaluate(ds, self.energy_eV())
        text = render_xs(ds, res)
        payload = {"xs": asdict(res)}
        if self.mode == "sample" and res.supported:
            n, seed = self.sampling()
            s = core.sample_outcomes(res, n, seed)
            text += "\n\n" + render_sample(res, s)
            payload["sample"] = s
        return RunOutput(text, res.supported, json.dumps(payload, indent=2), res)


# --------------------------------------------------------------------------- widgets
if tk is not None:

    class Page(ttk.Frame):
        title = ""
        subtitle = ""
        next_label = "Next >"

        def __init__(self, parent, model: WizardModel, app: "WizardApp"):
            super().__init__(parent, padding=(18, 8))
            self.model, self.app = model, app
            self.build()

        def build(self): ...
        def on_show(self): ...
        def validate(self) -> str | None:
            return None

    class WelcomePage(Page):
        title = "Welcome"
        subtitle = "An educational model of neutron-nucleus reactions"

        def build(self):
            banner = ttk.Label(self, text=BANNER.replace("=", "").strip(), wraplength=560,
                               font=self.app.fonts["bold"])
            banner.pack(anchor="w", fill="x", pady=(0, 10))
            autowrap(banner)
            msg = ("This wizard walks you through choosing a target isotope and a neutron energy, "
                   "then shows how likely each kind of interaction is according to evaluated "
                   "nuclear data (ENDF/B-VIII.0).\n\n"
                   "Results are probabilities. A random sample illustrates a distribution; it does "
                   "not predict what any real neutron will do. Chain reactions, criticality, yield "
                   "and material handling are out of scope by design.")
            body = ttk.Label(self, text=msg, wraplength=560, justify="left")
            body.pack(anchor="w", fill="x")
            autowrap(body)
            self.var = tk.BooleanVar(value=self.model.acknowledged)
            ttk.Checkbutton(self, variable=self.var, command=self._sync,
                            text="I understand this is an educational model only.").pack(anchor="w", pady=16)

        def _sync(self):
            self.model.acknowledged = self.var.get()
            self.app.update_buttons()

        def validate(self):
            return self.model.check_acknowledged()

    class IsotopePage(Page):
        title = "Target isotope"
        subtitle = "Only isotopes with processed evaluated data are offered"

        def build(self):
            self.names = supported_isotopes()
            ttk.Label(self, text="Select the nucleus the neutron will hit:").pack(anchor="w")
            self.lb = tk.Listbox(self, height=max(2, min(len(self.names), 8)), exportselection=False,
                                 font=self.app.fonts["mono_big"], activestyle="none")
            for n in self.names:
                self.lb.insert("end", n)
            self.lb.pack(anchor="w", pady=8)
            self.lb.bind("<<ListboxSelect>>", self._select)
            self.info = tk.StringVar(value="Loading details for the selected isotope appears here.")
            info = ttk.Label(self, textvariable=self.info, wraplength=560, justify="left")
            info.pack(anchor="w", fill="x")
            autowrap(info)
            self.note = ttk.Label(self, text="Anything else (periodic-table button): no evaluated data, so only a "
                                  "labelled model estimate is available.", foreground=MUTED,
                                  wraplength=560)
            self.note.pack(anchor="w", fill="x", pady=(10, 0))
            autowrap(self.note)
            ttk.Button(self, text="Explore any isotope (periodic table)...", command=self.app.open_explorer).pack(anchor="w", pady=(10, 0))
            if self.model.isotope in self.names:
                i = self.names.index(self.model.isotope)
                self.lb.selection_set(i)
                self._show(self.model.isotope)
            elif self.model.isotope:
                self.choose(self.model.isotope)

        def choose(self, iso: str):
            """Select `iso` programmatically (used by the explorer window). Any valid isotope is accepted: with
            evaluated data it is listed; without, it is carried forward as a labelled model estimate."""
            self.lb.selection_clear(0, "end")
            self.model.isotope = iso
            if iso in self.names:
                self.lb.selection_set(self.names.index(iso))
                self._show(iso)
            else:
                self.info.set(f"{iso}: no evaluated data in this project.\nThe wizard will run a MODEL ESTIMATE "
                              "from physics laws (rough, with wide ranges), clearly labelled as such. "
                              "Press Next to choose the energy.")
            self.app.update_buttons()

        def _select(self, _e=None):
            sel = self.lb.curselection()
            if sel:
                self.model.isotope = self.names[sel[0]]
                self._show(self.model.isotope)
                self.app.update_buttons()

        def _show(self, iso: str):
            self.info.set("Reading data file and checking its SHA-256 ...")
            self.app.run_bg(lambda: isotope_info(iso), lambda inf: self._fill(iso, inf))

        def _fill(self, iso, inf):
            if iso != self.model.isotope:
                return  # selection changed while loading
            if isinstance(inf, Exception):
                self.info.set(f"Could not load {iso}: {inf}")
                return
            self.info.set(f"{inf.library}; evaluated energy range {inf.e_min_eV:g} eV to "
                          f"{inf.e_max_eV:g} eV.\nChannels: " + ", ".join(f"MT{m}" for m in inf.mts))

        def validate(self):
            return self.model.check_isotope()

    class EnergyPage(Page):
        title = "Neutron energy"
        subtitle = "Required: the same isotope behaves differently at different energies"

        def build(self):
            row = ttk.Frame(self)
            row.pack(anchor="w")
            self.e_var = tk.StringVar(value=self.model.energy_text)
            self.u_var = tk.StringVar(value=self.model.unit)
            ttk.Label(row, text="Energy:").pack(side="left")
            ent = ttk.Entry(row, textvariable=self.e_var, width=14)
            ent.pack(side="left", padx=6)
            ttk.Combobox(row, textvariable=self.u_var, values=list(core.UNITS), width=6,
                         state="readonly").pack(side="left")
            ttk.Label(self, text="Quick picks:").pack(anchor="w", pady=(14, 2))
            pr = ttk.Frame(self)
            pr.pack(anchor="w")
            for i, (label, val, unit) in enumerate(PRESET_ENERGIES):   # grid: wraps to 2 columns on narrow windows
                ttk.Button(pr, text=label, command=lambda v=val, u=unit: self._preset(v, u)).grid(
                    row=i // 2, column=i % 2, padx=(0, 6), pady=2, sticky="w")
            self.hint = tk.StringVar()
            hint = ttk.Label(self, textvariable=self.hint, wraplength=560, justify="left", foreground=MUTED)
            hint.pack(anchor="w", fill="x", pady=(16, 0))
            autowrap(hint)
            self.err = tk.StringVar()
            err = ttk.Label(self, textvariable=self.err, foreground=ERROR, wraplength=560)
            err.pack(anchor="w", fill="x", pady=6)
            autowrap(err)
            for v in (self.e_var, self.u_var):
                v.trace_add("write", lambda *_: self._changed())

        def _preset(self, val, unit):
            self.e_var.set(val)
            self.u_var.set(unit)

        def _changed(self):
            self.model.energy_text, self.model.unit = self.e_var.get(), self.u_var.get()
            self.err.set("")
            self.app.update_buttons()

        def on_show(self):
            if not self.model.has_data:
                self.hint.set(f"{self.model.isotope} has no evaluated data: a model estimate can be run from "
                              f"{ESTIMATE_E_MIN_eV:g} eV to {ESTIMATE_E_MAX_eV:g} eV (above 1 keV: optical model + Hauser-Feshbach, low "
                              "confidence, checked against data for four isotopes only).")
                return
            try:
                inf = isotope_info(self.model.isotope)
                self.hint.set(f"{self.model.isotope} data cover {inf.e_min_eV:g} eV to {inf.e_max_eV:g} eV "
                              "(0.0253 eV is the conventional thermal energy).")
            except Exception:
                self.hint.set("")

        def validate(self):
            msg = self.model.check_energy()
            self.err.set(msg or "")
            return msg

    class ModePage(Page):
        title = "What to compute"
        subtitle = "Cross sections are always shown; random draws are optional"

        def build(self):
            self.mode = tk.StringVar(value=self.model.mode)
            self.n = tk.StringVar(value=self.model.n_text)
            self.seed = tk.StringVar(value=self.model.seed_text)
            ttk.Radiobutton(self, text="Cross sections and interaction probabilities", value="xs",
                            variable=self.mode, command=self._changed).pack(anchor="w", pady=2)
            ttk.Radiobutton(self, text="Also draw random outcomes from those probabilities", value="sample",
                            variable=self.mode, command=self._changed).pack(anchor="w", pady=2)
            self.opts = ttk.Frame(self)
            self.opts.pack(anchor="w", padx=24, pady=8)
            ttk.Label(self.opts, text="Number of draws:").grid(row=0, column=0, sticky="w")
            self.n_entry = ttk.Entry(self.opts, textvariable=self.n, width=12)
            self.n_entry.grid(row=0, column=1, padx=6, pady=2)
            ttk.Label(self.opts, text="Random seed (optional):").grid(row=1, column=0, sticky="w")
            self.s_entry = ttk.Entry(self.opts, textvariable=self.seed, width=12)
            self.s_entry.grid(row=1, column=1, padx=6, pady=2)
            ttk.Label(self.opts, text="Same seed = same draws, for repeatable examples.",
                      foreground=MUTED).grid(row=2, column=0, columnspan=2, sticky="w")
            self.err = tk.StringVar()
            err = ttk.Label(self, textvariable=self.err, foreground=ERROR, wraplength=560)
            err.pack(anchor="w", fill="x")
            autowrap(err)
            for v in (self.n, self.seed):
                v.trace_add("write", lambda *_: self._changed())
            self._changed()

        def _changed(self):
            self.model.mode, self.model.n_text, self.model.seed_text = (
                self.mode.get(), self.n.get(), self.seed.get())
            state = "normal" if self.mode.get() == "sample" else "disabled"
            for w in (self.n_entry, self.s_entry):
                w.configure(state=state)
            self.err.set("")
            self.app.update_buttons()

        def validate(self):
            msg = self.model.check_mode()
            self.err.set(msg or "")
            return msg

    class ReviewPage(Page):
        title = "Review"
        subtitle = "Check your choices, then run"
        next_label = "Run"

        def build(self):
            self.text = tk.StringVar()
            ttk.Label(self, textvariable=self.text, font=self.app.fonts["mono_big"], justify="left").pack(
                anchor="w", pady=10)   # fixed-width lines: kept unwrapped so the columns stay aligned
            ttk.Label(self, text="Use Back to change anything. Nothing is written to disk.",
                      foreground=MUTED).pack(anchor="w")

        def on_show(self):
            try:
                self.text.set(self.model.summary())
            except core.InputError as exc:
                self.text.set(str(exc))

    class ResultsPage(Page):
        title = "Results"
        subtitle = "Probabilities from evaluated data, with sources and assumptions"
        next_label = "Close"

        def build(self):
            bar = ttk.Frame(self)
            bar.pack(fill="x")
            ttk.Button(bar, text="Copy", command=self.copy).pack(side="left")
            ttk.Button(bar, text="Save text...", command=lambda: self.save(False)).pack(side="left", padx=6)
            ttk.Button(bar, text="Save JSON...", command=lambda: self.save(True)).pack(side="left")
            self.vbtn = ttk.Button(bar, text="Visualize...", command=self.visualize)
            self.vbtn.state(["disabled"])
            self.vbtn.pack(side="left", padx=6)
            self.bbtn = ttk.Button(bar, text="Box test...", command=self.box_test)
            self.bbtn.pack(side="left")
            ttk.Button(bar, text="Start over", command=self.app.restart).pack(side="right")
            box = ttk.Frame(self)
            box.pack(fill="both", expand=True, pady=(8, 0))
            self.out = tk.Text(box, wrap="none", font=self.app.fonts["mono"], width=20, height=8)  # small request; it grows with the window
            ys = ttk.Scrollbar(box, command=self.out.yview)
            xs = ttk.Scrollbar(box, orient="horizontal", command=self.out.xview)
            self.out.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
            ys.pack(side="right", fill="y")
            xs.pack(side="bottom", fill="x")
            self.out.pack(side="left", fill="both", expand=True)
            self.result: RunOutput | None = None

        def box_test(self):
            self.app.open_box()

        def visualize(self):
            if self.result and self.result.result is not None:
                _viz().open_window(self.app, self.result.result)

        def on_show(self):
            self.result = None
            self._set("Running ...\n(first load of a data file checks its SHA-256; U-235 takes a few seconds)")
            self.app.run_bg(self.model.run, self._done)

        def _set(self, text):
            self.out.configure(state="normal")
            self.out.delete("1.0", "end")
            self.out.insert("1.0", text)
            self.out.configure(state="disabled")

        def _done(self, r):
            if isinstance(r, core.InputError | core.UnsupportedError):
                self._set(f"error: {r}")
            elif isinstance(r, core.DataIntegrityError):
                self._set(f"data integrity error: {r}")
            elif isinstance(r, Exception):
                self._set(f"unexpected error: {r!r}")
            else:
                self.result = r
                self._set(r.text)
                if r.supported or r.result is not None:      # a model-estimate simulation can be animated too
                    self.vbtn.state(["!disabled"])

        def copy(self):
            self.clipboard_clear()
            self.clipboard_append(self.out.get("1.0", "end"))

        def save(self, as_json: bool):
            if not self.result:
                return
            path = filedialog.asksaveasfilename(
                defaultextension=".json" if as_json else ".txt",
                initialfile=f"neutronsim_{self.model.isotope}.{'json' if as_json else 'txt'}")
            if path:
                Path(path).write_text(self.result.json_text if as_json else self.result.text, encoding="utf-8")

    class ScrollFrame(ttk.Frame):
        """A frame whose contents (`.inner`) fill it when there is room and scroll when there is not.

        The vertical scrollbar only appears when the contents are taller than the window, so on a
        normal screen nothing changes; on a small screen or at a large font zoom nothing is cut off.
        """

        def __init__(self, parent):
            super().__init__(parent)
            try:
                bg = ttk.Style(self).lookup("TFrame", "background") or None
            except Exception:
                bg = None
            self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, **({"bg": bg} if bg else {}))
            self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
            self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
            self.canvas.pack(side="left", fill="both", expand=True)
            self._hbar_shown = False
            self.inner = ttk.Frame(self.canvas)
            self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
            self._bar_shown = False
            self.inner.bind("<Configure>", lambda _e: self._layout())
            self.canvas.bind("<Configure>", lambda _e: self._layout())
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.bind_all(seq, self._wheel, add="+")

        def _layout(self):
            try:
                cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
                need_h, need_w = self.inner.winfo_reqheight(), self.inner.winfo_reqwidth()
                h, w = max(ch, need_h), max(cw, need_w)
                self.canvas.itemconfigure(self._win, width=w, height=h)
                self.canvas.configure(scrollregion=(0, 0, w, h))
                show_v, show_h = need_h > ch + 1, need_w > cw + 1
                if show_v != self._bar_shown:
                    self._bar_shown = show_v
                    if show_v:
                        self.vbar.pack(side="right", fill="y", before=self.canvas)
                    else:
                        self.vbar.pack_forget()
                        self.canvas.yview_moveto(0)
                if show_h != self._hbar_shown:
                    self._hbar_shown = show_h
                    if show_h:
                        self.hbar.pack(side="bottom", fill="x", before=self.canvas)
                    else:
                        self.hbar.pack_forget()
                        self.canvas.xview_moveto(0)
            except tk.TclError:
                pass  # widget is being destroyed

        def _wheel(self, event):
            if not self._bar_shown:
                return
            try:
                w = event.widget
                if isinstance(w, tk.Text | tk.Listbox) or w.winfo_toplevel() is not self.winfo_toplevel():
                    return  # those scroll themselves / belong to another window
                step = -1 if (event.num == 4 or getattr(event, "delta", 0) > 0) else 1
                self.canvas.yview_scroll(step, "units")
            except (tk.TclError, AttributeError):
                pass

    class WizardApp(tk.Tk):
        PAGES = [WelcomePage, IsotopePage, EnergyPage, ModePage, ReviewPage, ResultsPage]

        def __init__(self):
            super().__init__()
            self.title("neutronsim setup wizard")
            self._make_fonts()
            fit_window(self, 680, 560, min_w=200, min_h=150)   # shrinks to fit small screens, resizable
            self.model = WizardModel()
            self.index = 0
            self._q: queue.Queue = queue.Queue()
            self._build_chrome()
            self._menu()
            self.show(0)
            self._front()
            self.after(100, self._poll)

        def _make_fonts(self):
            d, f = tkfont.nametofont("TkDefaultFont"), tkfont.nametofont("TkFixedFont")
            size, mono = abs(int(d.cget("size"))) or 12, abs(int(f.cget("size"))) or 12

            def mk(base, **kw):
                x = base.copy()
                x.configure(**kw)
                return x
            self.fonts = {"title": mk(d, size=size + 5, weight="bold"), "bold": mk(d, weight="bold"),
                          "mono": mk(f, size=mono), "mono_big": mk(f, size=mono + 1)}
            # base sizes, so Ctrl +/- zoom can scale every font in the window from a fixed reference
            self._zoom = 0
            self._base_sizes = {"title": size + 5, "bold": size, "mono": mono, "mono_big": mono + 1}
            self._std_fonts = {n: (tkfont.nametofont(n), abs(int(tkfont.nametofont(n).cget("size"))) or 12)
                               for n in ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont")}

        def zoom(self, step: int):
            """Ctrl +/-: make all text larger or smaller (step 0 resets). The window re-lays itself out."""
            self._zoom = 0 if step == 0 else max(-4, min(12, self._zoom + step))
            for key, font in self.fonts.items():
                font.configure(size=max(7, self._base_sizes[key] + self._zoom))
            for font, base in self._std_fonts.values():
                font.configure(size=max(7, base + self._zoom))

        def _front(self):
            """Terminal-launched Tk windows on macOS often open behind other windows."""
            self.lift()
            self.attributes("-topmost", True)
            self.after(200, lambda: self.attributes("-topmost", False))
            self.focus_force()

        # --- layout
        def _build_chrome(self):
            head = ttk.Frame(self, padding=(18, 12, 18, 4))
            head.pack(fill="x")
            self.h_title = ttk.Label(head, font=self.fonts["title"])
            self.h_title.pack(anchor="w")
            self.h_sub = ttk.Label(head, foreground=MUTED, wraplength=560)
            self.h_sub.pack(anchor="w", fill="x")
            autowrap(self.h_sub, pad=4)
            self.steps = ttk.Label(head, foreground=MUTED, wraplength=560)
            self.steps.pack(anchor="w", fill="x", pady=(6, 0))
            autowrap(self.steps, pad=4)
            ttk.Separator(self).pack(fill="x", pady=4)
            foot = ttk.Frame(self, padding=(18, 4, 18, 12))
            foot.pack(side="bottom", fill="x")
            ttk.Separator(self).pack(side="bottom", fill="x", pady=4)
            self.msg = tk.StringVar()
            msg = ttk.Label(self, textvariable=self.msg, foreground=ERROR, padding=(18, 0), wraplength=560)
            msg.pack(side="bottom", anchor="w", fill="x")
            autowrap(msg, pad=40)
            self.body = ScrollFrame(self)
            self.body.pack(fill="both", expand=True)
            # buttons first, so a narrow window shortens the disclaimer instead of pushing them off-screen
            self.next_btn = ttk.Button(foot, command=self.next)
            self.next_btn.pack(side="right")
            self.back_btn = ttk.Button(foot, text="< Back", command=self.back)
            self.back_btn.pack(side="right", padx=6)
            disclaimer = ttk.Label(foot, text="Educational model - not for laboratory, reactor, or weapons design.",
                                   foreground=MUTED, wraplength=400)
            disclaimer.pack(side="left", fill="x", expand=True)
            autowrap(disclaimer, pad=8, min_wrap=100)
            for seq in ("<Control-plus>", "<Control-equal>", "<Control-KP_Add>"):
                self.bind(seq, lambda _e: self.zoom(+1))
            for seq in ("<Control-minus>", "<Control-KP_Subtract>"):
                self.bind(seq, lambda _e: self.zoom(-1))
            self.bind("<Control-Key-0>", lambda _e: self.zoom(0))
            self.bind("<Return>", lambda _e: self.next_btn.instate(["!disabled"]) and self.next())

        def _menu(self):
            m = tk.Menu(self, tearoff=0)
            h = tk.Menu(m, tearoff=0, name="help")
            h.add_command(label="Concepts...", command=self.concepts)
            v = tk.Menu(m, tearoff=0)
            v.add_command(label="Larger text   (Ctrl +)", command=lambda: self.zoom(+1))
            v.add_command(label="Smaller text  (Ctrl -)", command=lambda: self.zoom(-1))
            v.add_command(label="Reset text size (Ctrl 0)", command=lambda: self.zoom(0))
            m.add_cascade(label="View", menu=v)
            t = tk.Menu(m, tearoff=0)
            t.add_command(label="Box test (neutrons inside a block)...", command=self.open_box)
            m.add_cascade(label="Tools", menu=t)
            m.add_cascade(label="Help", menu=h)
            self.config(menu=m)

        def concepts(self):
            win = tk.Toplevel(self)
            win.title("Concepts")
            fit_window(win, 700, 560, min_w=360, min_h=240)
            t = tk.Text(win, wrap="word", width=40, height=10, padx=10, pady=10)   # small request: the window decides
            sb = ttk.Scrollbar(win, command=t.yview)
            t.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
            t.pack(side="left", fill="both", expand=True)
            for k, v in notes.TOPICS.items():
                t.insert("end", f"{k.upper()}\n", "h")
                t.insert("end", v + "\n\n")
            t.tag_configure("h", font=self.fonts["bold"])
            t.configure(state="disabled")

        # --- navigation
        def open_box(self):
            """Box test window; starts from the wizard's isotope and energy where the box test supports them."""
            iso, e = self.model.isotope or "B-10", None
            try:
                e = self.model.energy_eV()
            except Exception:
                pass
            self.box_window = _boxgui().open_window(self, iso, e)

        def open_explorer(self):
            mod = _explorer()
            model = mod.ExplorerModel()
            if self.model.isotope:
                model.set_from_text(self.model.isotope)
                model.message = ""
            self.explorer = mod.ExplorerWindow(self, model, on_choose=self._explorer_choice)

        def _explorer_choice(self, iso: str):
            if isinstance(self.page, IsotopePage):
                self.page.choose(iso)

        def show(self, i: int):
            self.index = i
            for w in self.body.inner.winfo_children():
                w.destroy()
            self._building = True          # pages may poke update_buttons() while still being built
            try:
                page = self.PAGES[i](self.body.inner, self.model, self)
            finally:
                self._building = False
            page.pack(fill="both", expand=True)
            self.page = page
            self.h_title.config(text=page.title)
            self.h_sub.config(text=page.subtitle)
            names = [p.title for p in self.PAGES]
            self.steps.config(text="  >  ".join(f"[{n}]" if j == i else n for j, n in enumerate(names)))
            self.msg.set("")
            page.on_show()
            self.update_buttons()

        def update_buttons(self):
            if getattr(self, "_building", False):
                return
            last = self.index == len(self.PAGES) - 1
            self.next_btn.config(text=self.page.next_label)
            self.back_btn.state(["disabled"] if self.index == 0 or last else ["!disabled"])
            ok = True
            if isinstance(self.page, WelcomePage):
                ok = self.model.acknowledged
            elif isinstance(self.page, IsotopePage):
                ok = bool(self.model.isotope)
            self.next_btn.state(["!disabled"] if ok else ["disabled"])

        def next(self):
            if self.index == len(self.PAGES) - 1:
                self.destroy()
                return
            msg = self.page.validate()
            self.msg.set(msg or "")
            if msg:
                return
            self.show(self.index + 1)

        def back(self):
            if self.index > 0:
                self.show(self.index - 1)

        def restart(self):
            self.model = WizardModel(acknowledged=True)
            self.show(1)

        # --- background work so the window never freezes while data load
        def run_bg(self, fn, done):
            def work():
                try:
                    r = fn()
                except Exception as exc:  # delivered to the UI thread and shown there
                    r = exc
                self._q.put((done, r))
            threading.Thread(target=work, daemon=True).start()

        def _poll(self):
            try:
                while True:
                    done, r = self._q.get_nowait()
                    try:
                        done(r)
                    except tk.TclError:
                        pass  # the page that asked was closed meanwhile; drop its result
            except queue.Empty:
                pass
            finally:
                self.after(100, self._poll)


def main() -> int:
    if tk is None:
        print("error: tkinter is not available in this Python (install python3-tk, or use the CLI: "
              "'neutronsim xs ...').", file=__import__("sys").stderr)
        return 2
    try:
        app = WizardApp()
    except tk.TclError as exc:  # e.g. no display
        print(f"error: cannot open a window ({exc}). Use the CLI instead.", file=__import__("sys").stderr)
        return 2
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
