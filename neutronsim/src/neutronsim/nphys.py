"""Step 6: neutron-nucleus forecast for ANY nuclide from physics laws (stdlib only).

What is computed, and from which law:
  * nuclear size            R = 1.35 A^(1/3) fm
  * low-energy elastic      potential scattering, sigma = 4 pi R^2
  * fast-neutron reaction   black-disk / geometric cross section, sigma_nonel = pi (R + lambda-bar)^2, with
                            lambda-bar = 1/k from the neutron's de Broglie wavelength (kinematics in the centre of mass);
                            shadow (diffraction) elastic scattering is about equal to it
  * radiative capture      compound-nucleus statistics: Bohr compound-nucleus picture + average resonance theory
                            sigma_avg = 2 pi^2 lambda-bar^2 S0 sqrt(E) Gamma_g/(Gamma_g+Gamma_n)   (S0 = s-wave strength function, 1e-4)
                            and, between resonances (thermal), the Breit-Wigner tail  ~ 0.53 (Gamma_g/D) x the average.
                            Gamma_g/D comes from a back-shifted Fermi-gas level density (a = A/8) and a Brink-Axel giant-dipole
                            radiative strength whose size is fixed by the Thomas-Reiche-Kuhn sum rule (60 NZ/A mb MeV x 1.2).
                            The level-density parameter a = A/10 MeV^-1 is the top of the usual A/8..A/10 range; it was chosen
                            because it reproduces the measured s-wave level spacing of U-238 (model 20 eV vs ~21 eV), an observable
                            independent of the capture data being predicted. (It misses Au-197 by ~7x: reported by check().)
Compound-nucleus energies use the semi-empirical mass formula of physics.py, so everything inherits its few-MeV error.

HONESTY (all measured in check(), against real ENDF data and against 34 thermal capture values typed from memory):
  * One global constant C = calibration()[0] is fitted to the thermal capture values: the raw statistical estimate is systematically
    low (by a factor of ~20). It is a calibration, not a derivation; it is printed wherever it is used and the raw value is kept.
    It leaves a trend with mass number (heavy nuclei overpredicted): check() prints the slope.
  * Capture of ONE isotope at thermal energy is set by the nearest resonance, which is effectively random: the measured scatter
    after calibration is a factor of ~13 (68%). A shown range is that scatter, not a rigorous bound.
  * The resonance-average capture formula needs many resonances inside the averaging window. Where the model's own level spacing says
    they are dense it matched window-averaged ENDF within 0.18-2.6x (Ag-109, U-235); where sparse (B-10, Al-27) the formula does not
    apply, the claimed range widens to x/÷25 and the observed ratios were 0.09-0.88.
  * Coverage statistics for the thermal capture set are IN-SAMPLE: the same 34 values set the calibration and the scatter.
  * Closed-shell nuclides (Z or N = 2, 8, 20, 28, 50, 82, 126) are flagged: the model has no shell effects and overpredicts Pb-208
    capture by ~3x10^5.
  * 1 keV - 20 MeV: optical model + Hauser-Feshbach-type decay (hf.py), a simplified model: Weisskopf-Ewing limit (no angular-momentum
    coupling), estimated discrete levels, no width fluctuations, no pre-equilibrium or fission. Checked against ENDF for 4 isotopes only
    (hf_validation, IN-SAMPLE); each claimed factor is the worst observed miss rounded up. Where the compound nucleus is too light or weakly
    bound the old smooth estimates (potential elastic, capture below 100 keV) remain. Above 20 MeV the geometric estimate is used.
  * No fission cross sections, neutron yields or chain-reaction quantities, by design (project scope).
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from functools import lru_cache

from . import physics
from .nuclide import Nuclide

HBARC = 197.3269804           # MeV fm
R0_FM = 1.35                  # fm, nuclear radius parameter
S0 = 1.0e-4                   # s-wave neutron strength function (dimensionless; typical, varies 0.5-5e-4 with A)
KAPPA_TAIL = 0.53             # Breit-Wigner tail vs resonance average for equally spaced levels (derived in the docstring text)
SPIN_FACTOR = 1.5             # two s-wave spin states vs one in Gamma_g/D
MAGIC = (2, 8, 20, 28, 50, 82, 126)
E_FAST_MIN = 5e6              # eV: black-disk regime starts here (validated 5-14 MeV)
E_LOW_MAX = 1e3               # eV: potential scattering regime ends here
E_CAPTURE_MAX = 1e5           # eV: average-resonance capture formula not used above this
LEVEL_A_DIV = 10.0            # level-density parameter a = A / LEVEL_A_DIV (see docstring)
ELASTIC_FACTOR = 2.0          # potential scattering below 1 keV (observed 0.51-1.52 on window-averaged ENDF)
INTERMEDIATE_ELASTIC_FACTOR = 3.5   # potential scattering 1 keV - 5 MeV (observed 0.32-1.52)
FAST_FACTOR = 1.5             # geometric nonelastic and total at >= 5 MeV (observed ~0.7-1.45)
FAST_ELASTIC_FACTOR = 1.7     # shadow elastic at >= 5 MeV (observed ~0.62-1.2)
DENSE_RATIO = 20              # 'dense' resonances: model spacing D <= E / 20
CAPTURE_DENSE_FACTOR = 6.0    # resonance-average capture, dense resonances (observed 0.18-2.6)
CAPTURE_SPARSE_FACTOR = 25.0  # resonance-average capture, sparse resonances (observed 9-20 too high; the formula is not valid there)
HF_FACTORS = {                    # claimed multiplicative scatter of the Hauser-Feshbach rows: set from check() (see its summary), not guessed
    "el": 6.0, "tot": 6.0, "nonel": 6.0, "inel": 6.0, "cap": 30.0, "chg": 8.0, "multi": 3.0, "light": 30.0,
}
CLOSED_SHELL_EXTRA = 1e5          # extra widening for closed-shell capture: Pb-208 is overpredicted ~3e5 (model 134 b, real 4.9e-4 b); with the scatter factor the range now covers it
MIN_SN = 1.0                  # MeV: below this the compound nucleus is barely bound; capture is not predicted


# --------------------------------------------------------------------------- kinematics and sizes
def radius_fm(a: int) -> float:
    return R0_FM * a ** (1 / 3)


def lambda_bar_fm(e_lab_eV: float, a: int) -> float:
    """Reduced de Broglie wavelength 1/k in the centre-of-mass frame (fm)."""
    if e_lab_eV <= 0:
        raise ValueError("neutron energy must be positive")
    mu = physics.M_N * a * 931.49410242 / (physics.M_N + a * 931.49410242)       # reduced mass, MeV
    e_cm = e_lab_eV * 1e-6 * (a * 931.49410242) / (physics.M_N + a * 931.49410242)   # nonrelativistic E_cm, MeV
    k = math.sqrt(2 * mu * e_cm) / HBARC
    return 1 / k


def potential_scattering_b(a: int) -> float:
    return 4 * math.pi * radius_fm(a) ** 2 / 100.0


def geometric_nonelastic_b(a: int, e_lab_eV: float) -> float:
    return math.pi * (radius_fm(a) + lambda_bar_fm(e_lab_eV, a)) ** 2 / 100.0


def is_closed_shell(z: int, n: int) -> bool:
    """Target or compound nucleus with a magic proton or neutron number."""
    return z in MAGIC or n in MAGIC or (n + 1) in MAGIC


# --------------------------------------------------------------------------- level density and radiative strength
def _rho(a: int, z: int, n: int, e_ex: float) -> tuple[float, float]:
    """Back-shifted Fermi-gas total level density (per MeV) and spin cutoff sigma^2 at excitation e_ex (MeV)."""
    ad = a / LEVEL_A_DIV
    chi = 1 if (z % 2 == 0 and n % 2 == 0) else (-1 if (z % 2 and n % 2) else 0)
    u = max(e_ex - chi * 12 / math.sqrt(a), 0.3)
    s2 = 0.0888 * a ** (2 / 3) * math.sqrt(ad * u)
    return math.exp(2 * math.sqrt(ad * u)) / (12 * math.sqrt(2) * math.sqrt(s2) * ad ** 0.25 * u ** 1.25), s2


def _f_e1(eg: float, a: int, z: int, n: int) -> float:
    """Brink-Axel E1 radiative strength (MeV^-3) from a Lorentzian giant dipole resonance; size from the TRK sum rule."""
    e_g = 31.2 * a ** (-1 / 3) + 20.6 * a ** (-1 / 6)
    g_g = 0.026 * e_g ** 1.91
    s_g = 1.2 * 120 * n * z / (math.pi * a * g_g)                      # mb, peak cross section
    sig = s_g * (eg * g_g) ** 2 / ((eg ** 2 - e_g ** 2) ** 2 + (eg * g_g) ** 2)
    return 8.674e-8 * sig / eg


@lru_cache(maxsize=None)
def gamma_over_spacing(z: int, n: int) -> float | None:
    """Gamma_gamma / D for s-wave capture on target (z, n); None if the compound nucleus is too weakly bound for the model."""
    sn = physics.separation_energies(z, n + 1)[0]
    if sn is None or sn < MIN_SN or z + n + 1 < 4:
        return None
    ac = z + n + 1
    steps, tot = 400, 0.0
    for i in range(1, steps):
        eg = sn * i / steps
        tot += _f_e1(eg, ac, z, n + 1) * eg ** 3 * _rho(ac, z, n + 1, sn - eg)[0] * (sn / steps)
    return tot / (2 * math.pi) * SPIN_FACTOR


def level_spacing_eV(z: int, n: int) -> float | None:
    """Rough s-wave level spacing D (eV): only used to place the boundary between 'tail' and 'resonance-average' regimes."""
    sn = physics.separation_energies(z, n + 1)[0]
    if sn is None or sn < MIN_SN:
        return None
    ac = z + n + 1
    rho, s2 = _rho(ac, z, n + 1, sn)
    two_i1 = 1 if (z % 2 == 0 and n % 2 == 0) else (4 if ac % 2 == 0 else 5)       # assumed target spin: 0 / 3/2 / 2
    return s2 / (two_i1 * rho) * 1e6


def capture_raw_b(z: int, n: int, e_lab_eV: float, regime: str = "tail") -> float | None:
    """Uncalibrated statistical (n,gamma) estimate. regime 'tail': between resonances (thermal 1/v); 'average': energy-averaged."""
    g_d = gamma_over_spacing(z, n)
    if g_d is None:
        return None
    a = z + n
    l2 = lambda_bar_fm(e_lab_eV, a) ** 2 / 100.0                          # barn
    avg = 2 * math.pi ** 2 * l2 * S0 * math.sqrt(e_lab_eV)
    if regime == "tail":
        return KAPPA_TAIL * g_d * avg
    return avg * g_d / (g_d + S0 * math.sqrt(e_lab_eV))                   # Gamma_g/(Gamma_g+Gamma_n)


# --------------------------------------------------------------------------- calibration from the reference set
@lru_cache(maxsize=1)
def _thermal_set():
    try:
        d = json.loads(physics.REFERENCE_FILE.read_text())["thermal_capture_barn"]["values"]
    except (OSError, ValueError, KeyError):
        return ()
    out = []
    for k, v in d.items():
        nu = Nuclide.parse(k)
        raw = capture_raw_b(nu.Z, nu.N, 0.0253)
        out.append((k, nu, v, raw, is_closed_shell(nu.Z, nu.N)))
    return tuple(out)


@lru_cache(maxsize=1)
def calibration() -> tuple[float, float, int]:
    """(C, F, n): C = median(real/raw) over non-closed-shell reference nuclides; F = 10^(sd of log10(real/(C raw))); n = count."""
    rows = [(v, raw) for _k, _nu, v, raw, shell in _thermal_set() if raw and not shell]
    if len(rows) < 5:
        return 1.0, 100.0, 0
    logs = [math.log10(v / raw) for v, raw in rows]
    c_log = statistics.median(logs)
    sd = statistics.pstdev([x - c_log for x in logs])
    return 10 ** c_log, 10 ** sd, len(rows)


# --------------------------------------------------------------------------- the forecast
@dataclass
class Row:
    name: str
    value_b: float | None            # None = not predicted at this energy
    factor: float | None = None      # multiplicative uncertainty (68%-ish); range = value/factor .. value*factor
    note: str = ""


@dataclass
class Forecast:
    key: str
    energy_eV: float
    regime: str
    rows: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    s_n: float | None = None
    closed_shell: bool = False
    channels: dict = field(default_factory=dict)     # MT -> Row for every channel with a cross section (set where hf.py applies)


def _hf():
    from . import hf          # imported late: hf.py itself imports this module
    return hf


_HF_LABELS = {102: "capture (n,gamma)", 4: "inelastic (n,n')"}


def _hf_rows(fc: Forecast, nuc: Nuclide, e_eV: float) -> None:
    """1 keV - 20 MeV: optical-model cross sections and Hauser-Feshbach channel split from hf.py."""
    hf = _hf()
    z, n, a = nuc.Z, nuc.N, nuc.A
    res = hf.decay(z, n, e_eV)
    cs = hf.channel_sigmas(res)
    light = a < physics.LIGHT_A
    fac = (lambda k: HF_FACTORS["light"]) if light else (lambda k: HF_FACTORS[k])
    fc.regime = ("intermediate (optical model + Hauser-Feshbach; LOW CONFIDENCE)" if e_eV < E_FAST_MIN\
                 else "fast (optical model + Hauser-Feshbach; LOW CONFIDENCE)")
    how = "optical model (Koning-Delaroche-type) + statistical decay of the compound nucleus"
    el = Row("elastic (optical model + compound)", res.elastic_b, fac("el"), f"shape {res.sigma_shape_b:.3g} b + compound elastic")
    fc.rows.append(el)
    fc.channels[2] = el
    order = [4, 102] + sorted(mt for mt in cs if mt not in (4, 102))
    for mt in order:
        v = cs.get(mt, 0.0)
        if mt not in (4, 102) and v < 1e-9:
            continue
        label = _HF_LABELS.get(mt) or hf.LABEL_OF.get(mt, "other multi-particle channels")
        kind = "inel" if mt == 4 else ("cap" if mt == 102 else ("chg" if mt in (103, 107, 111, 112) else "multi"))
        f = fac(kind) * (30 if (kind == "cap" and fc.closed_shell) else 1)
        if v < 1e-12:
            row = Row(label, None, None, "closed or negligible at this energy")
        else:
            row = Row(label, v, f, "energy-averaged; " + ("no resonance structure" if mt in (4, 102) else "evaporation only: direct and pre-equilibrium parts missing"))
        fc.rows.append(row)
        if row.value_b is not None:
            fc.channels[mt] = row
    nonel = res.sigma_cn_b * (1 - res.compound_elastic)          # compound nucleus that does NOT go back out as an elastic neutron
    fc.rows.append(Row("nonelastic (all reactions together)", nonel, fac("nonel"), how))
    fc.rows.append(Row("total", res.sigma_total_b, fac("tot"), "optical model"))
    th = ", ".join(f"{k} {v:.2g}" for k, v in hf.thresholds_mev(z, n).items())
    fc.notes.append(f"model channel thresholds (MeV, model masses: rms error about 0.8 MeV on 15 known thresholds): {th}")
    if (z + n) % 2 == 1 or z % 2 == 1:
        fc.notes.append("odd nucleus: the first excited level (hence the inelastic threshold) is uncertain by about a factor 7 (rms 0.84 dex on 15 known nuclides)")
    fc.notes.append("discrete levels are ESTIMATED from the nucleus type (no level scheme is stored); angular-momentum and parity coupling, "
                    "width fluctuations, pre-equilibrium emission and fission are not modelled")
    if light:
        fc.notes.append(f"A < {physics.LIGHT_A}: shell and cluster structure are missing; the claimed ranges are very wide")


def forecast(nuc: Nuclide, e_eV: float) -> Forecast:
    if e_eV <= 0:
        raise ValueError("neutron energy must be positive")
    z, n, a = nuc.Z, nuc.N, nuc.A
    shell = is_closed_shell(z, n)
    fc = Forecast(nuc.key, e_eV, "", closed_shell=shell)
    sn = physics.separation_energies(z, n + 1)[0]
    fc.s_n = sn
    if a < 4:
        fc.notes.append("nuclei this light are outside what the model can describe")
    if shell:
        fc.notes.append("closed-shell nuclide: the model has no shell effects; capture estimates can be off by orders of magnitude "
                        "(Pb-208 is overpredicted by ~3x10^5)")
    c, f, _cnt = calibration()
    pot = potential_scattering_b(a)
    if e_eV <= E_LOW_MAX:
        fc.regime = "low energy (potential scattering + capture)"
        fc.rows.append(Row("elastic (potential scattering)", pot, ELASTIC_FACTOR, "4 pi R^2; the free-atom value at thermal energy"))
        d = level_spacing_eV(z, n)
        regime = "tail" if (d is None or e_eV < d / 2) else "average"
        raw = capture_raw_b(z, n, e_eV, regime)
        if raw is None:
            fc.rows.append(Row("capture (n,gamma)", None, None, "compound nucleus too weakly bound / too light for the model"))
        elif regime == "tail":
            fc.rows.append(Row("capture (n,gamma)", c * raw, f * (CLOSED_SHELL_EXTRA if shell else 1),
                               f"between-resonance 1/v tail x calibration {c:.2f} (raw {raw:.3g} b)"))
        else:
            dense = d is not None and d <= e_eV / DENSE_RATIO
            fc.rows.append(Row("capture (n,gamma)", raw, CAPTURE_DENSE_FACTOR if dense else CAPTURE_SPARSE_FACTOR,
                               "resonance-averaged (the real curve has resonances); " +
                               ("many resonances in the window: formula applies" if dense else
                                "few resonances in the window: formula is NOT reliable here")))
        fc.rows.append(Row("inelastic and other reactions", None, None, "none open / not predicted at this energy"))
    elif e_eV <= _hf().HF_MAX_E_eV and _hf().available(z, n) is None:
        _hf_rows(fc, nuc, e_eV)
    elif e_eV < E_FAST_MIN:
        fc.regime = "intermediate (LOW CONFIDENCE)"
        fc.rows.append(Row("elastic (potential scattering)", pot, INTERMEDIATE_ELASTIC_FACTOR,
                           "smooth trend only; real data has resonances and compound elastic"))
        if e_eV <= E_CAPTURE_MAX:
            raw = capture_raw_b(z, n, e_eV, "average")
            d = level_spacing_eV(z, n)
            dense = d is not None and d <= e_eV / DENSE_RATIO
            fc.rows.append(Row("capture (n,gamma)", raw, CAPTURE_DENSE_FACTOR if dense else CAPTURE_SPARSE_FACTOR,
                               "resonance-averaged; " + ("many resonances in the window" if dense else "few resonances: NOT reliable"))
                           if raw else Row("capture (n,gamma)", None, None, "compound nucleus too weakly bound / too light"))
        else:
            fc.rows.append(Row("capture (n,gamma)", None, None, "not modelled above 100 keV"))
        fc.rows.append(Row("inelastic and other reactions", None, None,
                           "NOT predicted here: the compound nucleus is too light or too weakly bound for the Hauser-Feshbach model"))
    else:
        fc.regime = "fast (geometric / black-disk)"
        nonel = geometric_nonelastic_b(a, e_eV)
        fc.rows.append(Row("nonelastic (all reactions together)", nonel, FAST_FACTOR, "pi (R + lambda-bar)^2"))
        fc.rows.append(Row("elastic (shadow scattering)", nonel, FAST_ELASTIC_FACTOR, "about equal to nonelastic for a black disk"))
        fc.rows.append(Row("total", 2 * nonel, FAST_FACTOR, ""))
        fc.rows.append(Row("capture (n,gamma)", None, None, "tiny at these energies; not modelled"))
    if sn is not None:
        fc.notes.append(f"capture would release about {sn:.1f} MeV of gamma energy (model S_n of {nuc.symbol}-{a + 1}, error up to ~3 MeV)")
    if n + 1 >= 0 and z + n + 1 <= 350:
        est = physics.estimate(z, n + 1)
        ch = ", ".join(f"{c} (Q {q:+.1f} MeV)" for c, q, _m in est.allowed[:3]) or "none"
        fc.notes.append(f"after capture: {nuc.symbol}-{a + 1} (model verdict {est.verdict.upper()}; energetically allowed decays: {ch}); "
                        "decay speed is not modelled")
    return fc


def render(fc: Forecast) -> list[str]:
    from .compare import _fmt_e
    out = [f"FORECAST for a neutron of {_fmt_e(fc.energy_eV)} hitting {fc.key}  [{fc.regime}]",
           "  MODEL ESTIMATE from physics laws, not measured data. Ranges are measured scatter against known cases, not guarantees."]
    for r in fc.rows:
        if r.value_b is None:
            out.append(f"  {r.name:<38} not predicted  ({r.note})")
        else:
            rng = "" if r.factor is None else f"  range {r.value_b / r.factor:.3g} - {r.value_b * r.factor:.3g}"
            out.append(f"  {r.name:<38}{r.value_b:>11.4g} b{rng}  ({r.note})" if r.note else
                       f"  {r.name:<38}{r.value_b:>11.4g} b{rng}")
    out += [f"  note: {t}" for t in fc.notes]
    out.append("  Not modelled: fission, neutron multiplicity, chain reactions (out of scope by design), gamma spectra, half-lives.")
    return out


# --------------------------------------------------------------------------- how wrong is it?
ENDF_ISOTOPES = ("B-10", "Al-27", "Ag-109", "U-235")
D0_REFERENCE_eV = {"U-238": 20.9, "Au-197": 16.0}   # measured s-wave spacings, typed from memory (approximate)


def _window_average(ds, e_eV: float, mt: int, span: float = 3.0, points: int = 61) -> float:
    """Average of an ENDF channel over e/span .. e*span (log-spaced): what an energy-averaged model should be compared with."""
    from . import core
    vals = []
    for i in range(points):
        e = e_eV * span ** (-1 + 2 * i / (points - 1))
        res = core.evaluate(ds, e)
        vals.append(res.total.sigma_barn if mt == 1 else
                    next((c.sigma_barn for c in res.channels if getattr(c, "mt", None) == mt), 0.0))
    return sum(vals) / len(vals)


# --------------------------------------------------------------------------- validation of the optical-model + Hauser-Feshbach rows
HF_ENERGIES_eV = (1e4, 1e5, 1e6, 2e6, 5e6, 14e6)


def hf_validation() -> tuple[list[str], dict]:
    """Model (hf.py) vs ENDF for B-10, Al-27, Ag-109, U-235 at 10 keV, 100 keV, 1, 2, 5 and 14 MeV. Below 5 MeV the data are averaged over
    e/2 .. 2e (resonances); at 5 and 14 MeV the point value is used. For U-235 the model has no fission, so its inelastic row is compared with
    (n,n') + (n,fission) of the data: the compound nucleus the model splits is the one that would also fission. Groups with A < 20 (B-10) are kept apart:
    the liquid-drop level density and the missing cluster structure make the model unreliable there and its misses must not set the claim for heavier nuclei."""
    from . import core
    out = ["Optical model + Hauser-Feshbach (hf.py) vs ENDF, ratio model/data (data averaged over e/2..2e below 5 MeV; point value from 5 MeV):"]
    groups = ("el", "tot", "nonel", "inel", "cap", "chg", "multi")
    st = {"main": {g: [] for g in groups}, "light": {g: [] for g in groups}, "missed_threshold": [], "false_open": [], "n_cases": 0}
    for iso in ENDF_ISOTOPES:
        nu = Nuclide.parse(iso)
        ds = core.Dataset(iso)
        key = "light" if nu.A < physics.LIGHT_A else "main"
        for e in HF_ENERGIES_eV:
            fc = forecast(nu, e)
            rows = {r.name: r for r in fc.rows}
            if e < 5e6:
                av = lambda mt: _window_average(ds, e, mt, span=2.0, points=41)
                d = {mt: av(mt) for mt in (1, 2, 4, 16, 17, 18, 102, 103, 107, 22, 28)}
            else:
                res = core.evaluate(ds, e)
                dd = {c.mt: c.sigma_barn for c in res.channels if hasattr(c, "mt")}
                d = {mt: dd.get(mt, 0.0) for mt in (2, 4, 16, 17, 18, 102, 103, 107, 22, 28)}
                d[1] = res.total.sigma_barn
            m = lambda mt: fc.channels[mt].value_b if mt in fc.channels else 0.0
            fissile = nu.Z >= 90
            pairs = [("el", "elastic", m(2), d[2]), ("tot", "total", rows["total"].value_b, d[1]),
                     ("nonel", "nonelastic", rows["nonelastic (all reactions together)"].value_b, d[1] - d[2]),
                     ("inel", "inelastic" + (" (+fission)" if nu.Z >= 90 else ""), m(4), d[4] + (d[18] if nu.Z >= 90 else 0.0)),
                     ("cap", "capture", m(102), d[102]), ("chg", "(n,p)+(n,a)", m(103) + m(107), d[103] + d[107]),
                     ("multi", "(n,2n)+(n,3n)+(n,np)+(n,na)", m(16) + m(17) + m(28) + m(22), d[16] + d[17] + d[28] + d[22])]
            line = []
            for g, name, mv, dv in pairs:
                if dv < 1e-4 and mv < 1e-4:
                    continue
                if fissile and ((g == "inel" and not 1e6 <= e <= 5e6) or g == "multi"):
                    continue          # fission takes most of the compound nucleus there and the model has none: not a fair comparison
                st["n_cases"] += 1
                if mv <= 0 or mv < 1e-9:
                    st["missed_threshold"].append((iso, e, name, dv))
                    line.append(f"{name} model closed (data {dv:.3g})")
                elif dv < 1e-4:
                    st["false_open"].append((iso, e, name, mv))
                    line.append(f"{name} model {mv:.3g}, data ~0")
                else:
                    st[key][g].append(mv / dv)
                    line.append(f"{name} {mv / dv:.2f}")
            out.append(f"  {iso:<7}{e:>9.3g} eV  " + "; ".join(line))
    out += ["", "Observed ratios model/data (heavier nuclides: Al-27, Ag-109, U-235) next to the factor each row claims "
            "(U-235: (n,n') compared with (n,n')+(n,fission) only at 1-5 MeV, no multi-particle comparison; fission is not in the model):"]
    names = {"el": "elastic", "tot": "total", "nonel": "nonelastic", "inel": "inelastic (n,n')", "cap": "capture", "chg": "(n,p)+(n,alpha)",
             "multi": "multi-particle (n,2n) ..."}
    for g in groups:
        v = st["main"][g]
        rng = f"{min(v):.2f}-{max(v):.2f} ({len(v)} cases)" if v else "no case with data and model both open"
        lt = st["light"][g]
        ltx = f"; B-10: {min(lt):.2g}-{max(lt):.2g}" if lt else ""
        out.append(f"  {names[g]:<26}{rng}  (claims x/÷{HF_FACTORS[g]:g}{ltx}; A<20 claims x/÷{HF_FACTORS['light']:g})")
    mt, fo = st["missed_threshold"], st["false_open"]
    out.append(f"  channels the model leaves closed although the data show them (>1e-4 b): {len(mt)}; channels the model opens that the data show as ~0: {len(fo)}")
    out.append(f"  Only {len(ENDF_ISOTOPES)} isotopes at {len(HF_ENERGIES_eV)} energies, and the discrete-level rule, the optical-model parameters and the factors "
               "were chosen or recalled before looking at these data, but the factors printed above were then set from them: IN-SAMPLE, a thin check, not a proof.")
    return out, st


# Out-of-sample check at 14-14.8 MeV for nuclides that are NOT among the ENDF isotopes the factors were set from. Barn, typed from memory of
# standard evaluated / measured values (good to about 10-15 %), so a ratio is only meaningful to that level.
OOS_14MEV = {
    "Al-27": {"nonel": 0.98, "p": 0.074, "a": 0.115}, "Fe-56": {"nonel": 1.30, "n2n": 0.46, "p": 0.11, "a": 0.025},
    "Ni-58": {"nonel": 1.35, "p": 0.42}, "Cu-63": {"nonel": 1.45, "n2n": 0.55, "p": 0.07},
    "Au-197": {"nonel": 1.95, "n2n": 2.1}, "Pb-208": {"nonel": 2.50, "n2n": 2.1},
}


def hf_out_of_sample() -> tuple[list[str], dict[str, list[float]]]:
    """Model / recalled value at 14.5 MeV. Al-27 is also an ENDF isotope (its 14 MeV point is in hf_validation), the other five are new."""
    hf = _hf()
    out = ["Out-of-sample check at 14.5 MeV (recalled values, ~10-15 % good), ratio model/value:"]
    groups: dict[str, list[float]] = {"nonel": [], "multi": [], "chg": []}
    for iso, d in OOS_14MEV.items():
        nu = Nuclide.parse(iso)
        r = hf.decay(nu.Z, nu.N, 14.5e6)
        cs = hf.channel_sigmas(r)
        m = {"nonel": r.sigma_cn_b * (1 - r.compound_elastic), "n2n": cs.get(16, 0.0), "p": cs.get(103, 0.0), "a": cs.get(107, 0.0)}
        names = {"nonel": "nonelastic", "n2n": "(n,2n)", "p": "(n,p)", "a": "(n,alpha)"}
        grp = {"nonel": "nonel", "n2n": "multi", "p": "chg", "a": "chg"}
        parts = []
        for q, v in d.items():
            groups[grp[q]].append(m[q] / v)
            parts.append(f"{names[q]} {m[q] / v:.2f}")
        out.append(f"  {iso:<7} " + "; ".join(parts))
    return out, groups


def check() -> tuple[list[str], dict]:
    from . import core
    out = ["Forecast vs real ENDF/B-VIII.0 data for the 4 isotopes that have it (ratio = model / data):",
           "  (below 1 keV: potential scattering + capture; at 5 and 14 MeV the geometric estimate used only above 20 MeV; the 1 keV - 20 MeV model is checked further below)"]
    st = {k: [] for k in ("el_low", "nonel", "el_fast", "tot_fast", "cap_dense", "cap_sparse")}
    for iso in ENDF_ISOTOPES:
        nu = Nuclide.parse(iso)
        ds = core.Dataset(iso)
        for e in (1e3,):          # potential scattering + average-resonance capture; above 1 keV the hf.py rows are checked in hf_validation()
            el = _window_average(ds, e, 2)
            cap = _window_average(ds, e, 102)
            rows = {r.name: r for r in forecast(nu, e).rows}
            m_el = rows["elastic (potential scattering)"].value_b
            st["el_low"].append(m_el / el)
            out.append(f"  {iso:<7}{e:>9.3g} eV  elastic   model {m_el:8.3g}  data(avg) {el:8.3g}  ratio {m_el / el:5.2f}")
            cr = rows["capture (n,gamma)"]
            if cr.value_b and cap:
                d = level_spacing_eV(nu.Z, nu.N)
                dense = d is not None and d <= e / DENSE_RATIO
                st["cap_dense" if dense else "cap_sparse"].append(cr.value_b / cap)
                out.append(f"  {iso:<7}{e:>9.3g} eV  capture   model {cr.value_b:8.3g}  data(avg) {cap:8.3g}  ratio {cr.value_b / cap:5.2f}  "
                           f"({'dense' if dense else 'sparse'} resonances)")
        for e in (5e6, 14e6):          # geometric estimate = the fallback above 20 MeV; checked here on its validated 5-14 MeV range
            res = core.evaluate(ds, e)
            d = {c.mt: c.sigma_barn for c in res.channels if hasattr(c, "mt")}
            tot, el = res.total.sigma_barn, d.get(2, 0.0)
            non_g = geometric_nonelastic_b(nu.A, e)
            for key, name, m, data in (("nonel", "nonelastic", non_g, tot - el), ("el_fast", "elastic", non_g, el), ("tot_fast", "total", 2 * non_g, tot)):
                st[key].append(m / data)
                out.append(f"  {iso:<7}{e:>9.3g} eV  {name:<10} geometric {m:8.3g}  data {data:8.3g}  ratio {m / data:5.2f}")
    hf_lines, hf_st = hf_validation()
    from . import hf as _hf_mod
    lvl_lines, lvl_rms = _hf_mod.level_validation()
    oos_lines, oos = hf_out_of_sample()
    thr_lines, thr_rms = _hf_mod.threshold_validation()
    out += [""] + hf_lines + [""] + oos_lines + [""] + lvl_lines + thr_lines
    st["thr_rms"] = thr_rms
    st["oos"] = oos
    st["levels"] = lvl_rms
    st["hf"] = hf_st
    # level-density check on an observable that is not a capture cross section
    out += ["", "Level-density check (s-wave spacing D, model vs measured; measured typed from memory, approximate):"]
    st["spacing"] = {}
    for iso, real in D0_REFERENCE_eV.items():
        nu = Nuclide.parse(iso)
        m = level_spacing_eV(nu.Z, nu.N)
        st["spacing"][iso] = m / real
        out.append(f"  {iso:<7} model {m:8.3g} eV  measured ~{real:g} eV  ratio {m / real:5.2f}")
    c, f, cnt = calibration()
    out += ["", f"Thermal capture vs 34 reference values typed from memory (calibration C = {c:.1f} fitted on {cnt} non-closed-shell nuclides):",
            f"  {'iso':<8}{'real b':>10}{'raw b':>11}{'forecast b':>12}{'ratio':>9}  flag"]
    inside = total = 0
    xs, ys, worst_shell = [], [], 0.0
    for k, nu, v, raw, shell in _thermal_set():
        if raw is None:
            out.append(f"  {k:<8}{v:>10.4g}   not predicted (compound nucleus too light for the model)")
            continue
        m, r = raw * c, raw * c / v
        ok = 1 / f <= r <= f
        total += 1
        inside += ok
        if shell:
            worst_shell = max(worst_shell, abs(math.log10(r)))
        else:
            xs.append(math.log10(nu.A))
            ys.append(math.log10(r))
        out.append(f"  {k:<8}{v:>10.4g}{raw:>11.3g}{m:>12.3g}{r:>9.3g}  {'closed shell' if shell else ''}{'' if ok else '  OUTSIDE range'}")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    st.update(C=c, F=f, n_cal=cnt, within=inside, total=total, worst_shell=worst_shell, slope=slope)
    rng = lambda k: f"{min(st[k]):.2f}-{max(st[k]):.2f}"
    out += ["", "Summary of observed ratios (model/data) next to the factor each forecast claims:",
            f"  elastic below 1 keV      {rng('el_low')}  (claims x/÷{ELASTIC_FACTOR:g})",
            f"  geometric nonelastic, 5-14 MeV  {rng('nonel')}  (fallback above 20 MeV; claims x/÷{FAST_FACTOR:g})",
            f"  geometric elastic, 5-14 MeV     {rng('el_fast')}  (claims x/÷{FAST_ELASTIC_FACTOR:g})",
            f"  geometric total, 5-14 MeV       {rng('tot_fast')}  (claims x/÷{FAST_FACTOR:g})",
            f"  capture, dense resonances  {rng('cap_dense')}  (claims x/÷{CAPTURE_DENSE_FACTOR:g})",
            f"  capture, sparse resonances {rng('cap_sparse')}  (claims x/÷{CAPTURE_SPARSE_FACTOR:g}; formula not valid there)",
            f"  thermal capture: {inside}/{total} reference values inside x/÷{f:.1f} (in-sample); closed-shell worst miss 10^{worst_shell:.1f}; "
            f"residual trend with mass number {slope:+.2f} dex per dex of A (heavy nuclei overpredicted if positive)"]
    return out, st
