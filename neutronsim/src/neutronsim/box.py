"""Box test: neutrons released inside a block made of ONE isotope; how many are left, where does the energy go.

A cube of side L (vacuum outside) is filled uniformly with atoms of a single isotope at a chosen density.
Neutrons are released at the centre (or uniformly), each moves in straight lines between collisions, collides
with probability set by the evaluated total cross section, and the collision type is drawn from the evaluated
channel cross sections (the same DATA the rest of this project reads). A neutron ends by being absorbed
(capture, charged-particle reactions), by leaking out of the box, or by still being alive when the run stops.
Heat = all kinetic energy that leaves the neutrons plus the reaction Q-values, deposited where it happens.

SCOPE (same as the rest of the project): isotopes whose evaluated data contain a FISSION channel are refused.
A box of fissile material with a density knob and a heat counter is a chain-reaction / criticality calculation,
which this educational model excludes by design. Non-fissile isotopes (B-10, Al-27, Ag-109) are an attenuation /
absorption / heating problem: every neutron history is independent except for the rare (n,2n)-type births.

ASSUMPTIONS (printed with every result):
  * Cross sections: 0 K, target at rest, no Doppler broadening. Because targets are at rest, a neutron would
    slow down for ever; its energy is therefore never lowered below min(E_start_of_collision, 0.0253 eV).
  * Elastic scattering: isotropic in the centre-of-mass frame (exact kinematics), nucleus recoil energy = heat.
  * Neutron-emitting reactions other than elastic (inelastic, (n,2n), (n,np), ...): the data hold no emission
    spectra, so each emitted neutron is drawn from an evaporation spectrum E exp(-E/theta), theta =
    sqrt(E_available/a), a = A/10 per MeV, truncated to the energy that is available, direction isotropic
    in the lab. This is a MODEL, not data.
  * Everything that is not a neutron (gammas, charged particles, recoils) is assumed to stop INSIDE the box at
    the collision point. In a real block capture gammas partly escape, so real heating in a small box is lower.
  * Reaction Q-values: the evaluation's own value where stored; else, for endothermic threshold channels, the
    value implied by the threshold energy in the data; else a typed-from-memory reference value (B-10 only)
    or the semi-empirical model value (error up to a few MeV). The table is printed with its source per channel.
  * No neutron decay, no gravity, no neutron-neutron interaction, no thermal-scattering law, no crystal structure.
"""
from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass, field

from functools import lru_cache

from . import channels, compare, core, estsim, nphys, physics
from .nuclide import Nuclide

N_A = 6.02214076e23                 # 1/mol
U_PER_NEUTRON = 1.00866491595       # neutron mass in u (CODATA 2018); atom mass in u = awr * this
MN_C2_EV = 939.56542052e6           # neutron rest energy, eV
C_CM_S = 2.99792458e10              # cm/s
EV_TO_J = 1.602176634e-19
E_FLOOR_EV = 0.0253                 # energy below which scattering no longer lowers a neutron's energy (room temperature)
MAX_NEUTRONS = 100_000
MAX_SIDE_CM = 1000.0
MIN_SIDE_CM = 0.01
MAX_DENSITY_FACTOR = 2.0            # relative to the reference solid density
N_BINS = 100
T_MIN_S = 1e-9
FISSION_MTS = {18, 19, 20, 21, 38}

# Solid densities (g/cm3) TYPED FROM MEMORY (elemental values; enriched isotopes differ slightly). Not read from data.
REFERENCE_DENSITY = {"B-10": 2.34, "Al-27": 2.70, "Ag-109": 10.49}
# Reaction Q (MeV) typed from memory where neither the evaluation nor the threshold gives it:
Q_FROM_MEMORY = {("B-10", 107): 2.790, ("B-10", 103): 0.2304}
# Reference density by element Z (g/cm3), TYPED FROM MEMORY, room-temperature solid; for elements that are gases or liquids
# at room temperature the liquid/solid value near the boiling point. Only used to scale the density slider (1.0 = this value).
# Hydrogen..Radium; astatine (85) and francium (87) are left out on purpose (no value I trust).
ELEMENT_DENSITY = {
    1: 0.071, 2: 0.125, 3: 0.534, 4: 1.85, 5: 2.34, 6: 2.267, 7: 0.808, 8: 1.141, 9: 1.50, 10: 1.207, 11: 0.971, 12: 1.738,
    13: 2.70, 14: 2.33, 15: 1.82, 16: 2.07, 17: 1.56, 18: 1.396, 19: 0.862, 20: 1.55, 21: 2.985, 22: 4.506, 23: 6.11, 24: 7.19,
    25: 7.21, 26: 7.874, 27: 8.90, 28: 8.908, 29: 8.96, 30: 7.14, 31: 5.91, 32: 5.323, 33: 5.727, 34: 4.81, 35: 3.12, 36: 2.41,
    37: 1.532, 38: 2.64, 39: 4.472, 40: 6.52, 41: 8.57, 42: 10.28, 43: 11.0, 44: 12.37, 45: 12.41, 46: 12.02, 47: 10.49,
    48: 8.65, 49: 7.31, 50: 7.31, 51: 6.68, 52: 6.24, 53: 4.93, 54: 3.0, 55: 1.93, 56: 3.51, 57: 6.15, 58: 6.77, 59: 6.77,
    60: 7.01, 61: 7.26, 62: 7.52, 63: 5.24, 64: 7.90, 65: 8.23, 66: 8.55, 67: 8.80, 68: 9.07, 69: 9.32, 70: 6.90, 71: 9.84,
    72: 13.31, 73: 16.65, 74: 19.25, 75: 21.02, 76: 22.59, 77: 22.56, 78: 21.45, 79: 19.30, 80: 13.53, 81: 11.85, 82: 11.34,
    83: 9.78, 84: 9.20, 86: 4.4, 88: 5.5,
}
MODEL_MAX_Z = 88                    # actinides (Z >= 89) are refused on the model path: fission is not modelled
MODEL_E_MAX_eV = 2.0e7              # the physics-law forecast stops here (hf.py range)
GRID_PER_DECADE_LOW, GRID_PER_DECADE_HIGH = 4, 6
MAX_TRACES = 400                    # recorded neutron histories (the first ones, plus (n,2n)-type children)
MAX_TRACE_STEPS = 400               # recorded collisions per history

ASSUMPTIONS = [
    "Cross sections are 0 K data with the target at rest; no Doppler broadening, no thermal-scattering law, no crystal structure.",
    f"A neutron's energy is never lowered below min(its energy before the collision, {E_FLOOR_EV} eV) (targets at rest would otherwise slow it for ever).",
    "Elastic scattering: isotropic in the centre-of-mass frame, exact kinematics; the energy the neutron loses is heat.",
    "Neutron-emitting non-elastic reactions: emitted-neutron energies are drawn from an evaporation spectrum (a MODEL; the data hold no spectra), directions isotropic.",
    "All gammas, charged particles and recoils stop at the collision point (heat is local). Real capture gammas partly escape a small block.",
    "Reaction Q-values: evaluated where stored, else from the data's threshold, else typed-from-memory (B-10) or the semi-empirical model; see the Q table.",
    "Fission is out of scope: isotopes with a fission channel are refused. No chain reaction or criticality calculation is performed.",
]
MODEL_ASSUMPTIONS_BOX = [
    "MODEL ESTIMATE: this isotope has no evaluated data here, so every cross section comes from physics laws (nphys / hf.py: "
    "optical model + Hauser-Feshbach-type decay, potential scattering and compound-nucleus capture below 1 keV), tabulated on an "
    "energy grid and interpolated. No resonances, no shell effects; each cross section may be off by the factors printed by "
    "'neutronsim forecast ISOTOPE'.",
    "Reaction Q-values of a model isotope all come from the semi-empirical mass formula (error up to a few MeV); heat per capture "
    "inherits that error. Channel thresholds come from hf.py's own binding energies, so a channel can open slightly below or above "
    "the Q used here (the energy bookkeeping still closes).",
    "The target nucleus is treated as stable and is never changed by a reaction (no decay, no build-up of products, no activation).",
    "Reference solid density typed from memory (ELEMENT_DENSITY); it only sets what '1 x solid' means on the slider.",
]
DISCLAIMER = core.DISCLAIMER


@dataclass
class ChannelInfo:
    mt: int
    name: str
    n_out: int            # neutrons emitted by the reaction
    q_MeV: float
    q_source: str


@dataclass
class BoxResult:
    isotope: str
    energy_eV: float
    source: str
    side_cm: float
    density_g_cm3: float
    density_factor: float            # relative to REFERENCE_DENSITY
    reference_density_g_cm3: float
    atoms_per_cm3: float
    box_mass_g: float
    mean_free_path_cm: float         # at the starting energy, from the evaluated total cross section
    n_source: int
    n_born: int                      # extra neutrons created by (n,2n)-type reactions
    absorbed: int
    leaked: int
    alive_end: int                   # still moving when the run stopped (time limit or collision cap)
    stopped_by_cap: int              # part of alive_end: history hit the collision cap
    t_end_s: float
    collisions: int
    tally: dict                      # channel label -> number of collisions
    heat_MeV: float
    heat_J: float
    ke_leaked_MeV: float
    ke_alive_MeV: float
    energy_in_MeV: float             # source kinetic energy + sum of reaction Q
    closure_MeV: float               # energy_in - heat - leaked - alive  (should be ~0)
    q_table: list = field(default_factory=list)
    times_s: list = field(default_factory=list)       # right edge of each time bin
    alive_t: list = field(default_factory=list)
    absorbed_t: list = field(default_factory=list)
    leaked_t: list = field(default_factory=list)
    heat_MeV_t: list = field(default_factory=list)     # cumulative
    assumptions: list = field(default_factory=lambda: list(ASSUMPTIONS))
    disclaimer: str = DISCLAIMER
    model_estimate: bool = False                       # True: cross sections are physics-law estimates, not evaluated data
    traces: list = field(default_factory=list)         # recorded histories (Trace), only when run_box(trace=N) was asked

    @property
    def n_total(self) -> int:
        return self.n_source + self.n_born

    @property
    def heat_per_source_MeV(self) -> float:
        return self.heat_MeV / self.n_source


@dataclass
class Trace:
    """One neutron history, recorded for the animation. steps[i] = (x, y, z, E_eV_after, t_s, kind, mt) is where the i-th
    segment ends. kind: elastic | react (neutron-emitting reaction, history continues) | absorb | leak | alive (clock stopped) |
    cap (collision cap). start_phase: animation step at which it starts (children of an (n,2n) reaction start when the
    parent's reaction happens)."""
    start_phase: int
    x0: float
    y0: float
    z0: float
    E0: float
    steps: list = field(default_factory=list)
    end: str = ""                                      # kind of the last step, or "truncated"

    def add(self, x, y, z, E, t, kind, mt=None):
        if self.end:
            return
        if len(self.steps) >= MAX_TRACE_STEPS and kind in ("elastic", "react"):
            self.end = "truncated"
            return
        self.steps.append((x, y, z, E, t, kind, mt))
        if kind in ("absorb", "leak", "alive", "cap"):
            self.end = kind


def atoms_per_cm3(density_g_cm3: float, awr: float) -> float:
    return density_g_cm3 * N_A / (awr * U_PER_NEUTRON)


def _has_data(iso: str) -> bool:
    from . import explorer
    return iso in explorer.data_isotopes()


def reference_density(iso: str) -> float:
    """What '1 x solid' means for this isotope (g/cm3, typed from memory)."""
    if iso in REFERENCE_DENSITY:
        return REFERENCE_DENSITY[iso]
    z = Nuclide.parse(iso).Z
    if z not in ELEMENT_DENSITY:
        raise core.UnsupportedError(f"{iso}: no reference density is stored for element Z={z}")
    return ELEMENT_DENSITY[z]


def isotope_awr(iso: str) -> float:
    """Atomic mass / neutron mass (from the evaluated file where there is one, else the model mass)."""
    iso = core.normalize_isotope(iso)
    if _has_data(iso):
        return core.Dataset(iso).info["awr"]
    nuc = Nuclide.parse(iso)
    return physics.atomic_mass(nuc.Z, nuc.N) / physics.M_N


def refusal_reason(iso: str) -> str | None:
    """None if the box test runs for this isotope (any isotope except fissile / actinide ones), otherwise why not."""
    try:
        iso = core.normalize_isotope(iso)
        nuc = Nuclide.parse(iso)
    except Exception as exc:
        return str(exc)
    if _has_data(iso):
        ds = core.Dataset(iso)
        if FISSION_MTS & set(ds.tabs):
            return (f"{iso} has a fission channel in its data. A box of fissile material with a density control and a heat counter is a "
                    "chain-reaction / criticality calculation, which this educational model excludes by design (see README, "
                    "'Limitations'). Every other isotope works, for example " + ", ".join(boxable_isotopes()) + ".")
        return None
    if nuc.Z < 1:
        return f"{iso}: a box of free neutrons is not a material."
    if nuc.Z > MODEL_MAX_Z:
        return (f"{iso} is an actinide (Z >= 89). Fission is not modelled, so the result would leave out the dominant process, and a block of "
                "fissile-capable material with a density control and a heat counter is a chain-reaction / criticality calculation, which "
                "this educational model excludes by design. Every lighter isotope works.")
    if nuc.Z not in ELEMENT_DENSITY:
        return f"{iso}: no reference density is stored for element Z={nuc.Z}, so the density slider has no scale."
    return None


def boxable_isotopes() -> list[str]:
    from . import explorer
    out = []
    for iso in sorted(explorer.data_isotopes()):
        if not (FISSION_MTS & set(core.Dataset(iso).tabs)):
            out.append(iso)
    return out


class _GridTab:
    """A cross section tabulated on an energy grid, callable like a core cross-section table (log-log between positive points)."""
    def __init__(self, x: list[float], y: list[float]):
        self.x, self.y = x, y
        pos = [i for i, v in enumerate(y) if v > 0.0]
        self.e_min = x[pos[0]] if pos else x[-1]
        self.e_max = x[-1]
        self.qm = 0.0
        self._lx = [math.log(v) for v in x]

    def __call__(self, e: float) -> float:
        x, y = self.x, self.y
        if e <= x[0]:
            return y[0]
        if e >= x[-1]:
            return y[-1]
        i = bisect.bisect_right(x, e) - 1
        y0, y1 = y[i], y[i + 1]
        if y0 > 0.0 and y1 > 0.0:
            f = (math.log(e) - self._lx[i]) / (self._lx[i + 1] - self._lx[i])
            return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))
        return y0 + (y1 - y0) * (e - x[i]) / (x[i + 1] - x[i])


class ModelDataset:
    """Same interface the box engine reads from core.Dataset, filled from the physics-law forecast (estsim / nphys / hf)."""
    is_model = True

    def __init__(self, isotope: str, e_start_eV: float):
        self.isotope = isotope
        nuc = Nuclide.parse(isotope)
        self.nuc = nuc
        self.info = {"awr": physics.atomic_mass(nuc.Z, nuc.N) / physics.M_N}
        lo = min(e_start_eV, E_FLOOR_EV)
        self.energy_range_eV = (lo, MODEL_E_MAX_eV)
        grid = self._grid(lo, e_start_eV)
        cols: dict[int, list[float]] = {}
        for gi, e in enumerate(grid):
            parts = estsim._parts(nphys.forecast(nuc, e))      # raises UnsupportedError where the model cannot split the outcomes
            if any(mt == 3 for mt, _, _ in parts):
                raise core.UnsupportedError(
                    f"{isotope}: at {e:.4g} eV the model only knows 'elastic + non-elastic' (the nucleus is too light for the "
                    "Hauser-Feshbach model), so it cannot say which reaction happens. Start below 1 keV, or choose a heavier isotope.")
            for mt, _lab, row in parts:
                cols.setdefault(mt, [0.0] * len(grid))[gi] = float(row.value_b or 0.0)
        if 2 not in cols:
            raise core.UnsupportedError(f"{isotope}: the model gives no elastic cross section")
        self.tabs = {mt: _GridTab(grid, vals) for mt, vals in cols.items()}
        self.tabs[1] = _GridTab(grid, [sum(cols[mt][i] for mt in cols) for i in range(len(grid))])
        self.partial_mts = sorted(mt for mt in cols if mt != 2)
        self.grid = grid

    @staticmethod
    def _grid(lo: float, hi: float) -> list[float]:
        pts = {lo, hi}
        def span(a, b, per):
            if b <= a:
                return
            n = max(1, math.ceil(math.log10(b / a) * per))
            for k in range(n + 1):
                pts.add(a * (b / a) ** (k / n))
        span(lo, min(hi, 1e3), GRID_PER_DECADE_LOW)
        if hi > 1e3:
            pts.add(1e3)
            pts.add(1e3 * 1.001)
            span(1e3 * 1.001, hi, GRID_PER_DECADE_HIGH)
        return sorted(pts)


@lru_cache(maxsize=12)
def model_dataset(isotope: str, e_start_eV: float) -> ModelDataset:
    return ModelDataset(isotope, e_start_eV)


def _q_table(ds, e_max_eV: float) -> list[ChannelInfo]:
    """Q-values of the channels that can be open for neutrons of at most e_max_eV (neutrons never gain energy)."""
    nuc = Nuclide.parse(ds.isotope)
    A = ds.info["awr"]
    lo = ds.energy_range_eV[0]
    rows = []
    if getattr(ds, "is_model", False):
        for mt in ds.partial_mts:
            tab = ds.tabs[mt]
            if tab.e_min > e_max_eV:
                continue
            if mt == 4:
                n_out, q, src = 1, 0.0, "not applicable (level energy unknown; evaporation model)"
            else:
                em = compare.EMISSIONS.get(mt)
                q = compare.reaction_q(nuc.Z, nuc.N, mt)
                if em is None or q is None:
                    raise core.UnsupportedError(f"box test: channel MT{mt} of {ds.isotope} has no usable Q-value in the model; refusing rather than guessing")
                n_out, src = sum(1 for p in em if p == "n"), "MODEL (semi-empirical mass formula; error up to a few MeV)"
            rows.append(ChannelInfo(mt, estsim._label(mt), n_out, q, src))
        return rows
    for mt in ds.partial_mts:
        tab = ds.tabs[mt]
        if tab.e_min > e_max_eV:
            continue
        em = compare.EMISSIONS.get(mt)
        if mt == 4:
            n_out, q, src = 1, 0.0, "not applicable (level energy unknown; evaporation model)"
        elif mt == 102:
            n_out, q, src = 0, tab.qm / 1e6, "evaluation (stored)"
        elif em is None:
            raise core.UnsupportedError(f"box test: channel MT{mt} of {ds.isotope} has no emission bookkeeping; refusing rather than guessing")
        else:
            n_out = sum(1 for p in em if p == "n")
            model = compare.reaction_q(nuc.Z, nuc.N, mt)
            if tab.qm != 0.0:
                q, src = tab.qm / 1e6, "evaluation (stored)"
            elif (ds.isotope, mt) in Q_FROM_MEMORY:
                q, src = Q_FROM_MEMORY[(ds.isotope, mt)], "typed from memory (not read from data)"
            elif model is not None and model < 0 and tab.y[0] == 0.0 and tab.e_min > lo:
                q, src = -tab.e_min * A / (A + 1) / 1e6, "implied by the threshold energy in the data"
            elif model is not None:
                q, src = model, "MODEL (semi-empirical mass formula; error up to a few MeV)"
            else:
                raise core.UnsupportedError(f"box test: no Q-value available for MT{mt} of {ds.isotope}")
        rows.append(ChannelInfo(mt, channels.name(mt), n_out, q, src))
    return rows


def _evap(rng: random.Random, e_avail_eV: float, A: float) -> float:
    """One emitted-neutron energy (eV) from E exp(-E/theta), truncated to e_avail (MODEL)."""
    if e_avail_eV <= 0:
        return 0.0
    theta = math.sqrt((e_avail_eV / 1e6) * 10.0 / A) * 1e6       # eV
    for _ in range(200):
        e = -theta * (math.log(1.0 - rng.random()) + math.log(1.0 - rng.random()))
        if e <= e_avail_eV:
            return e
    return e_avail_eV * rng.random()


def speed_cm_s(e_eV: float) -> float:
    """Neutron speed (cm/s) from kinetic energy, relativistically exact: beta^2 = x(2+x)/(1+x)^2 with x = E/(m c^2).
    (Written this way so it stays accurate at 1e-5 eV; the non-relativistic formula is 1.6 % too fast at 20 MeV.)"""
    x = e_eV / MN_C2_EV
    return C_CM_S * math.sqrt(x * (2.0 + x)) / (1.0 + x)


def _isotropic(rng: random.Random):
    w = 2.0 * rng.random() - 1.0
    phi = 2.0 * math.pi * rng.random()
    s = math.sqrt(max(0.0, 1.0 - w * w))
    return s * math.cos(phi), s * math.sin(phi), w


def _rotate(u, v, w, mu, rng: random.Random):
    """New unit direction at lab cosine mu to (u,v,w), azimuth uniform."""
    phi = 2.0 * math.pi * rng.random()
    s = math.sqrt(max(0.0, 1.0 - mu * mu))
    cp, sp = math.cos(phi), math.sin(phi)
    if 1.0 - abs(w) < 1e-12:                 # exactly along the z axis (the general formula divides by sqrt(1-w^2))
        sg = 1.0 if w > 0 else -1.0
        return s * cp, s * sp * sg, mu * sg
    d = math.sqrt(1.0 - w * w)
    nu = mu * u + s * (u * w * cp - v * sp) / d
    nv = mu * v + s * (v * w * cp + u * sp) / d
    nw = mu * w - s * d * cp
    n = math.sqrt(nu * nu + nv * nv + nw * nw)
    return nu / n, nv / n, nw / n


def _validate(n, side_cm, t_end_s, max_collisions, source):
    if not isinstance(n, int) or not (1 <= n <= MAX_NEUTRONS):
        raise core.InputError(f"number of neutrons must be an integer between 1 and {MAX_NEUTRONS}")
    if not (MIN_SIDE_CM <= side_cm <= MAX_SIDE_CM) or not math.isfinite(side_cm):
        raise core.InputError(f"box side must be between {MIN_SIDE_CM} and {MAX_SIDE_CM} cm")
    if not (1e-8 <= t_end_s <= 1e3) or not math.isfinite(t_end_s):
        raise core.InputError("end time must be between 1e-8 and 1e3 seconds")
    if not isinstance(max_collisions, int) or max_collisions < 1:
        raise core.InputError("collision cap must be a positive integer")
    if source not in ("center", "uniform"):
        raise core.InputError("source must be 'center' or 'uniform'")


def run_box(isotope: str, e_eV: float, *, density_g_cm3: float | None = None, density_factor: float | None = None,
            side_cm: float = 10.0, n: int = 1000, seed: int | None = None, t_end_s: float = 0.1,
            source: str = "center", max_collisions: int = 100_000, dataset=None, trace: int = 0) -> BoxResult:
    """Run the box test. Give the density as g/cm3 OR as a factor of the reference solid density (default 1.0).
    Works for any isotope except fissile / actinide ones: evaluated data where the project has it, otherwise the
    physics-law model estimate. trace=N records the first N histories (plus their (n,2n)-type children) for the animation."""
    iso = core.normalize_isotope(isotope)
    why = refusal_reason(iso)
    if why:
        raise core.UnsupportedError(why)
    _validate(n, side_cm, t_end_s, max_collisions, source)
    is_model = dataset is None and not _has_data(iso)
    if is_model:
        if not (1e-5 <= e_eV <= MODEL_E_MAX_eV):
            raise core.UnsupportedError(f"model estimate: E={e_eV:g} eV is outside [1e-5, {MODEL_E_MAX_eV:g}] eV for {iso}")
        ds = model_dataset(iso, float(e_eV))
    else:
        ds = dataset or core.Dataset(iso)
    is_model = bool(getattr(ds, "is_model", False))
    lo, hi = ds.energy_range_eV
    if not (lo <= e_eV <= hi):
        raise core.UnsupportedError(f"unsupported by available data: E={e_eV:g} eV is outside [{lo:g}, {hi:g}] eV for {iso}")
    ref = reference_density(iso)
    if density_g_cm3 is not None and density_factor is not None:
        raise core.InputError("give the density either in g/cm3 or as a factor, not both")
    if density_g_cm3 is None:
        density_factor = 1.0 if density_factor is None else density_factor
        density_g_cm3 = density_factor * ref
    else:
        density_factor = density_g_cm3 / ref
    if not math.isfinite(density_g_cm3) or density_g_cm3 <= 0 or density_factor > MAX_DENSITY_FACTOR:
        raise core.InputError(f"density must be > 0 and at most {MAX_DENSITY_FACTOR} x the reference solid density ({ref * MAX_DENSITY_FACTOR:g} g/cm3)")

    A = ds.info["awr"]
    dens = atoms_per_cm3(density_g_cm3, A)
    qt = _q_table(ds, e_eV)
    tot, el = ds.tabs[1], ds.tabs[2]
    parts = [(ci, ds.tabs[ci.mt]) for ci in qt]
    labels = {2: "MT2 elastic scattering"}
    for ci in qt:
        labels[ci.mt] = f"MT{ci.mt} {ci.name}"
    tally = {lab: 0 for lab in labels.values()}

    rng = random.Random(seed)
    h = side_cm / 2.0
    lg_min = math.log10(T_MIN_S)
    step = (math.log10(t_end_s) - lg_min) / N_BINS

    def binof(t: float) -> int:
        if t <= T_MIN_S:
            return 0
        return min(N_BINS - 1, int((math.log10(t) - lg_min) / step))

    d_alive = [0] * N_BINS
    d_abs = [0] * N_BINS
    d_leak = [0] * N_BINS
    d_heat = [0.0] * N_BINS

    absorbed = leaked = born = collisions = capped = 0
    ke_leak = ke_alive = 0.0
    heat = 0.0
    q_sum = 0.0
    alive_end = 0
    a2p1, ap1sq = A * A + 1.0, (A + 1.0) ** 2

    traces: list[Trace] = []
    for hist in range(n):
        if source == "center":
            x = y = z = 0.0
        else:
            x, y, z = (rng.random() * side_cm - h for _ in range(3))
        root = Trace(0, x, y, z, e_eV) if hist < trace else None
        if root is not None:
            traces.append(root)
        stack = [(x, y, z, *_isotropic(rng), e_eV, 0.0, root)]
        while stack:
            x, y, z, u, v, w, E, t, tr = stack.pop()
            ncoll = 0
            while True:
                st = tot(E)
                if st <= 0:
                    st = 1e-30
                d = -math.log(1.0 - rng.random()) / (dens * st * 1e-24)
                # distance to the box surface along (u, v, w)
                dexit = math.inf
                if u > 0: dexit = min(dexit, (h - x) / u)
                elif u < 0: dexit = min(dexit, (-h - x) / u)
                if v > 0: dexit = min(dexit, (h - y) / v)
                elif v < 0: dexit = min(dexit, (-h - y) / v)
                if w > 0: dexit = min(dexit, (h - z) / w)
                elif w < 0: dexit = min(dexit, (-h - z) / w)
                speed = speed_cm_s(E)
                if d >= dexit:
                    t += dexit / speed
                    if t > t_end_s:                              # reached the wall only after the end time
                        alive_end += 1; ke_alive += E
                        if tr is not None:
                            f = speed * (t_end_s - (t - dexit / speed))
                            tr.add(x + u * f, y + v * f, z + w * f, E, t_end_s, "alive")
                        break
                    leaked += 1; ke_leak += E
                    if tr is not None:
                        tr.add(x + u * dexit, y + v * dexit, z + w * dexit, E, t, "leak")
                    b = binof(t); d_leak[b] += 1; d_alive[b] -= 1
                    break
                t += d / speed
                if t > t_end_s:
                    alive_end += 1; ke_alive += E
                    if tr is not None:
                        f = speed * (t_end_s - (t - d / speed))
                        tr.add(x + u * f, y + v * f, z + w * f, E, t_end_s, "alive")
                    break
                x += u * d; y += v * d; z += w * d
                ncoll += 1; collisions += 1
                if ncoll > max_collisions:
                    alive_end += 1; capped += 1; ke_alive += E
                    if tr is not None:
                        tr.add(x, y, z, E, t, "cap")
                    break
                # ---- choose the collision channel
                s_el = el(E)
                weights = [(None, s_el)]
                for ci, tab in parts:
                    if tab.e_min <= E <= tab.e_max:
                        s = tab(E)
                        if s > 0.0:
                            weights.append((ci, s))
                r = rng.random() * sum(s for _, s in weights)
                chosen = weights[-1][0]
                acc = 0.0
                for ci, s in weights:
                    acc += s
                    if r < acc:
                        chosen = ci
                        break
                b = binof(t)
                if chosen is None:                               # elastic, isotropic in the CM frame
                    tally[labels[2]] += 1
                    muc = 2.0 * rng.random() - 1.0
                    f = a2p1 + 2.0 * A * muc
                    En = max(E * f / ap1sq, min(E, E_FLOOR_EV))
                    mu_lab = (1.0 + A * muc) / math.sqrt(f)
                    q_heat = E - En
                    heat += q_heat; d_heat[b] += q_heat
                    # direction change uses the lab cosine of the unfloored kinematics
                    u, v, w = _rotate(u, v, w, max(-1.0, min(1.0, mu_lab)), rng)
                    E = En
                    if tr is not None:
                        tr.add(x, y, z, E, t, "elastic", 2)
                    continue
                tally[labels[chosen.mt]] += 1
                q_eV = chosen.q_MeV * 1e6
                q_sum += q_eV
                if chosen.n_out == 0:                            # absorbed: all energy becomes heat
                    heat_ev = E + q_eV
                    heat += heat_ev; d_heat[b] += heat_ev
                    absorbed += 1; d_abs[b] += 1; d_alive[b] -= 1
                    if tr is not None:
                        tr.add(x, y, z, 0.0, t, "absorb", chosen.mt)
                    break
                # neutron-emitting reaction: energy available to the emitted neutrons
                avail = max(0.0, A / (A + 1.0) * E + q_eV) if chosen.mt != 4 else A / (A + 1.0) * E
                outs = []
                remaining = avail
                for _k in range(chosen.n_out):
                    e_out = max(_evap(rng, remaining, A), lo)      # never below the data's lowest energy
                    remaining -= e_out
                    outs.append(e_out)
                heat_ev = E + q_eV - sum(outs)
                heat += heat_ev; d_heat[b] += heat_ev
                extra = chosen.n_out - 1
                if extra:
                    born += extra; d_alive[b] += extra
                if tr is not None:
                    tr.add(x, y, z, outs[0], t, "react", chosen.mt)
                for e_out in outs[1:]:
                    child = None
                    if tr is not None and len(traces) < MAX_TRACES:
                        child = Trace(tr.start_phase + len(tr.steps), x, y, z, e_out)
                        traces.append(child)
                    stack.append((x, y, z, *_isotropic(rng), e_out, t, child))
                u, v, w = _isotropic(rng)
                E = outs[0]

    # ---- bookkeeping
    e_in = n * e_eV + q_sum
    closure = e_in - heat - ke_leak - ke_alive
    cum_alive, cum_abs, cum_leak, cum_heat = n, 0, 0, 0.0
    times, a_t, ab_t, lk_t, h_t = [], [], [], [], []
    for i in range(N_BINS):
        cum_alive += d_alive[i]; cum_abs += d_abs[i]; cum_leak += d_leak[i]; cum_heat += d_heat[i]
        times.append(10 ** (lg_min + step * (i + 1)))
        a_t.append(cum_alive); ab_t.append(cum_abs); lk_t.append(cum_leak); h_t.append(cum_heat / 1e6)
    return BoxResult(
        isotope=iso, energy_eV=e_eV, source=source, side_cm=side_cm, density_g_cm3=density_g_cm3,
        density_factor=density_factor, reference_density_g_cm3=ref, atoms_per_cm3=dens,
        box_mass_g=density_g_cm3 * side_cm ** 3, mean_free_path_cm=1.0 / (dens * tot(e_eV) * 1e-24),
        n_source=n, n_born=born, absorbed=absorbed, leaked=leaked, alive_end=alive_end, stopped_by_cap=capped,
        t_end_s=t_end_s, collisions=collisions, tally=tally,
        heat_MeV=heat / 1e6, heat_J=heat * EV_TO_J, ke_leaked_MeV=ke_leak / 1e6, ke_alive_MeV=ke_alive / 1e6,
        energy_in_MeV=e_in / 1e6, closure_MeV=closure / 1e6,
        q_table=[(c.mt, c.name, c.q_MeV, c.q_source) for c in qt],
        times_s=times, alive_t=a_t, absorbed_t=ab_t, leaked_t=lk_t, heat_MeV_t=h_t,
        assumptions=list(ASSUMPTIONS) + (MODEL_ASSUMPTIONS_BOX if is_model else []), model_estimate=is_model, traces=traces)


def _fmt_e(e: float) -> str:
    return f"{e / 1e6:g} MeV" if e >= 1e6 else f"{e / 1e3:g} keV" if e >= 1e3 else f"{e:g} eV"


def render(r: BoxResult) -> str:
    pct = lambda k: 100.0 * k / r.n_total
    out = [
        f"BOX TEST  {r.isotope}  neutrons of {_fmt_e(r.energy_eV)}  ({r.source} source)"
        + ("   [MODEL ESTIMATE: no evaluated data, cross sections from physics laws]" if r.model_estimate else ""),
        f"  box: cube {r.side_cm:g} cm per side, {r.box_mass_g:g} g, density {r.density_g_cm3:.4g} g/cm3 "
        f"= {r.density_factor:.4g} x solid ({r.reference_density_g_cm3:g}, typed from memory), {r.atoms_per_cm3:.3e} atoms/cm3",
        f"  mean free path at the start energy ({'model-estimate' if r.model_estimate else 'evaluated'} total cross section): {r.mean_free_path_cm:.4g} cm",
        "",
        f"  neutrons released {r.n_source}" + (f", created by (n,2n)-type reactions {r.n_born}" if r.n_born else ""),
        f"  absorbed        {r.absorbed:8d}  ({pct(r.absorbed):6.2f} %)",
        f"  leaked out      {r.leaked:8d}  ({pct(r.leaked):6.2f} %)",
        f"  still alive     {r.alive_end:8d}  ({pct(r.alive_end):6.2f} %)  at t = {r.t_end_s:g} s"
        + (f"; {r.stopped_by_cap} stopped by the collision cap" if r.stopped_by_cap else ""),
        f"  collisions      {r.collisions:8d}  ({r.collisions / r.n_source:.3g} per released neutron)",
        "",
        f"  heat deposited in the box: {r.heat_MeV:.6g} MeV = {r.heat_J:.4g} J  ({r.heat_per_source_MeV:.4g} MeV per released neutron)",
        f"  energy carried out by leaked neutrons {r.ke_leaked_MeV:.6g} MeV; by neutrons still alive {r.ke_alive_MeV:.6g} MeV",
        f"  energy bookkeeping: in {r.energy_in_MeV:.6g} MeV (source + reaction Q) - heat - leaked - alive = {r.closure_MeV:.3g} MeV",
        "",
        "  collisions by type:",
    ]
    for lab, k in sorted(r.tally.items(), key=lambda kv: -kv[1]):
        if k:
            out.append(f"    {lab:<48s}{k:10d}")
    out.append("  Q-values used (MeV):")
    for mt, name, q, src in r.q_table:
        out.append(f"    MT{mt:<4d}{name:<44s}{q:+9.3f}  {src}")
    out.append("")
    out.append("  assumptions:")
    out += [f"    - {a}" for a in r.assumptions]
    out.append(f"  {r.disclaimer}")
    return "\n".join(out)
