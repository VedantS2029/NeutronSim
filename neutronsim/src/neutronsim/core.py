"""Core calculations. Every number returned carries a provenance record.

Three kinds of content are kept apart in every result:
  DATA        values read (and interpolated as stored) from an evaluated-data file
  CALCULATION values derived from DATA by a documented formula (e.g. probabilities)
  ASSUMPTION  model choices (target at rest, isolated nucleus, 0 K data, ...)
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import channels, endf

DATA_DIR = Path(os.environ.get("NEUTRONSIM_DATA", Path(__file__).resolve().parents[2] / "data"))
UNITS = {"eV": 1.0, "keV": 1e3, "MeV": 1e6}
REMAINDER_REL_TOL = 5e-3     # |total - elastic - sum(partials)| / total allowed before it is reported
MAX_SAMPLES = 1_000_000
LAW_NAMES = {1: "histogram", 2: "lin-lin", 3: "lin-log", 4: "log-lin", 5: "log-log"}

ASSUMPTIONS = [
    "Isolated target nucleus at rest (data are 0 K; no thermal motion, no chemical binding).",
    "Probabilities are conditional on an interaction having occurred; they say nothing about "
    "interaction rates, mean free paths, materials or geometry.",
    "Resonance-region values come from NJOY RECONR reconstruction of the evaluation's resonance "
    "parameters (err=0.001); unresolved-resonance self-shielding is not applied.",
]
DISCLAIMER = ("EDUCATIONAL MODEL - not a tool for laboratory, reactor, or weapons design. "
              "Outcomes are probabilities; no individual real event is predicted.")


class InputError(ValueError):
    """Invalid user input."""


class UnsupportedError(Exception):
    """Request outside what the available data can support."""


class DataIntegrityError(Exception):
    """A data file does not match its recorded SHA-256."""


def normalize_isotope(text: str) -> str:
    s = text.strip()
    m = re.fullmatch(r"([A-Za-z]{1,2})-?(\d{1,3})", s) or None
    if m:
        el, a = m.group(1), m.group(2)
    else:
        m = re.fullmatch(r"(\d{1,3})-?([A-Za-z]{1,2})", s)
        if not m:
            raise InputError(f"cannot parse isotope {text!r}; use forms like 'U-235', 'u235' or '235U'")
        a, el = m.group(1), m.group(2)
    return f"{el[0].upper()}{el[1:].lower()}-{int(a)}"


def to_eV(value: float, unit: str) -> float:
    if unit not in UNITS:
        raise InputError(f"unknown energy unit {unit!r}; choose from {', '.join(UNITS)}")
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"energy {value!r} is not a number") from exc
    if not math.isfinite(v) or v <= 0:
        raise InputError("neutron energy must be a finite number > 0")
    return v * UNITS[unit]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ChannelResult:
    mt: int
    name: str
    kind: str
    status: str                      # ok | below_threshold | unsupported
    sigma_barn: float | None
    probability: float | None = None
    note: str = ""
    uncertainty: str = "not propagated"
    source: dict | None = None       # provenance of the DATA value


@dataclass
class Result:
    isotope: str
    energy_eV: float
    supported: bool
    reason: str = ""
    total: ChannelResult | None = None
    channels: list[ChannelResult] = field(default_factory=list)
    remainder_barn: float | None = None
    remainder_fraction: float | None = None
    remainder_flag: str = ""
    assumptions: list[str] = field(default_factory=lambda: list(ASSUMPTIONS))
    disclaimer: str = DISCLAIMER


class Dataset:
    """One isotope's processed evaluated data plus its manifest record."""

    def __init__(self, isotope: str, data_dir: Path = DATA_DIR):
        manifest = json.loads((data_dir / "MANIFEST.json").read_text())
        if isotope not in manifest["isotopes"]:
            raise UnsupportedError(
                f"{isotope}: unsupported by available data. Supported: {', '.join(sorted(manifest['isotopes']))}")
        self.isotope = isotope
        self.info = manifest["isotopes"][isotope]
        self.processing = manifest["processing"]
        self.source = manifest["source"]
        path = data_dir / self.info["pendf_file"]
        got = _sha256(path)
        if got != self.info["pendf_sha256"]:
            raise DataIntegrityError(f"{path.name}: SHA-256 {got} != manifest {self.info['pendf_sha256']}")
        self.tabs = endf.read_mf3(path)
        self.library = self.info["library"]
        if 1 not in self.tabs or 2 not in self.tabs:
            raise endf.DataError(f"{isotope}: total or elastic section missing")

    @property
    def partial_mts(self) -> list[int]:
        return sorted(mt for mt in self.tabs if channels.is_summation_member(mt))

    @property
    def energy_range_eV(self) -> tuple[float, float]:
        t = self.tabs[1]
        return t.e_min, t.e_max

    def _processing_steps(self) -> list[str]:
        r = self.processing["reconr"]
        return [f"{self.processing['code']} RECONR (err={r['err']}, T={r['temperature_K']} K)",
                "extraction of MF3 sections (lines copied verbatim)"]

    def provenance(self, mt: int, e_eV: float | None) -> dict:
        t = self.tabs[mt]
        law = None
        if e_eV is not None and t.e_min <= e_eV <= t.e_max:
            i = min(max(0, bisect.bisect_right(t.x, e_eV) - 1), len(t.x) - 2)
            law = LAW_NAMES.get(t.law_at(i), str(t.law_at(i)))
        return {
            "kind": "DATA",
            "dataset": self.library,
            "evaluation": self.info["evaluation_title"],
            "isotope": self.isotope,
            "mat": self.info["mat"],
            "file": f"MF3/MT{mt}",
            "channel": channels.name(mt),
            "energy_range_eV": [t.e_min, t.e_max],
            "units": "barn",
            "interpolation": law,
            "source_file": self.info["raw_file"],
            "source_sha256": self.info["raw_sha256"],
            "processed_file": self.info["pendf_file"],
            "processed_sha256": self.info["pendf_sha256"],
            "processing": self._processing_steps(),
            "stored_QI_eV": t.qi,  # 0 may mean "not given" in the processed file
        }

    def uncertainty_note(self, mt: int) -> str:
        cov = self.info["covariance_sections_in_source"]["MF33 (cross sections)"]
        if mt in cov:
            return ("not propagated: the evaluation has a covariance (MF33) for this channel, "
                    "but this version does not read it")
        return "unavailable: no covariance for this channel in the source evaluation"

    def channel(self, mt: int, e_eV: float) -> ChannelResult:
        t = self.tabs[mt]
        nm, kd = channels.name(mt), channels.kind(mt)
        unc = self.uncertainty_note(mt)
        if e_eV < t.e_min:
            lo = self.energy_range_eV[0]
            if t.y[0] == 0.0 and t.e_min > lo:
                return ChannelResult(mt, nm, kd, "below_threshold", 0.0,
                                     note=(f"threshold-type table: first tabulated value (at {t.e_min:g} eV) is "
                                           "exactly 0; taken as zero below it (usual ENDF threshold convention). "
                                           "Cross-check: the summation rule still closes at this energy"),
                                     uncertainty=unc, source=self.provenance(mt, None))
            return ChannelResult(mt, nm, kd, "unsupported", None,
                                 note=f"unsupported by available data: E below stored range ({t.e_min:g} eV)",
                                 uncertainty=unc, source=self.provenance(mt, None))
        if e_eV > t.e_max:
            return ChannelResult(mt, nm, kd, "unsupported", None,
                                 note=f"unsupported by available data: E above stored range ({t.e_max:g} eV)",
                                 uncertainty=unc, source=self.provenance(mt, None))
        return ChannelResult(mt, nm, kd, "ok", t(e_eV), uncertainty=unc, source=self.provenance(mt, e_eV))


def evaluate(ds: Dataset, e_eV: float) -> Result:
    lo, hi = ds.energy_range_eV
    res = Result(ds.isotope, e_eV, supported=True)
    if not (lo <= e_eV <= hi):
        res.supported = False
        res.reason = (f"unsupported by available data: E={e_eV:g} eV is outside the evaluated range "
                      f"[{lo:g}, {hi:g}] eV for {ds.isotope} ({ds.library}). No extrapolation is performed.")
        return res

    total = ds.channel(1, e_eV)
    res.total = total
    rows = [ds.channel(2, e_eV)] + [ds.channel(mt, e_eV) for mt in ds.partial_mts]
    sig_t = total.sigma_barn
    if not sig_t or sig_t <= 0:
        res.supported = False
        res.reason = "total cross section is zero or missing at this energy"
        return res

    attributed = 0.0
    n_unsupported = 0
    for r in rows:
        if r.sigma_barn is None:
            n_unsupported += 1
            continue
        attributed += r.sigma_barn
        r.probability = r.sigma_barn / sig_t   # CALCULATION: sigma_i / sigma_total
    res.channels = rows
    res.remainder_barn = sig_t - attributed
    res.remainder_fraction = res.remainder_barn / sig_t
    if n_unsupported:
        res.remainder_flag = (f"{n_unsupported} channel(s) unsupported at this energy; the remainder "
                              "includes them")
    elif res.remainder_fraction < -REMAINDER_REL_TOL:
        res.remainder_flag = "INCONSISTENT: channel sum exceeds total beyond tolerance"
    elif res.remainder_fraction > REMAINDER_REL_TOL:
        res.remainder_flag = "channels not listed in this program's data account for the remainder"
    return res


def sample_outcomes(res: Result, n: int, seed: int | None = None) -> dict:
    """Draw n outcomes from the channel probabilities (CALCULATION, not data).

    Describes how outcomes would be distributed over many identical interactions; says nothing
    about what any single real event will do.
    """
    if not isinstance(n, int) or not (1 <= n <= MAX_SAMPLES):
        raise InputError(f"number of samples must be an integer between 1 and {MAX_SAMPLES}")
    if not res.supported:
        raise UnsupportedError(res.reason)
    if res.remainder_flag.startswith("INCONSISTENT"):
        raise UnsupportedError("cannot sample: " + res.remainder_flag)
    cats = [(f"MT{c.mt} {c.name}", c.probability) for c in res.channels if c.probability]
    rem_sig = abs(res.remainder_fraction) > REMAINDER_REL_TOL
    if rem_sig:
        cats.append(("unattributed remainder", res.remainder_fraction))
    norm = sum(p for _, p in cats)
    rng = random.Random(seed)
    labels = [c for c, _ in cats]
    counts = Counter(rng.choices(labels, weights=[p for _, p in cats], k=n))
    rows = []
    for lab, p in cats:
        pn = p / norm
        k = counts.get(lab, 0)
        rows.append({"outcome": lab, "model_probability": pn, "count": k, "observed_fraction": k / n,
                     "sampling_std_error": math.sqrt(pn * (1 - pn) / n)})
    return {
        "n": n, "seed": seed, "normalisation_factor": norm,
        "normalisation_note": ("probabilities divided by their sum; the factor differs from 1 only by "
                               "the within-tolerance remainder or by an included unattributed remainder"),
        "rows": rows,
        "error_meaning": ("sampling_std_error is Monte Carlo sampling noise only. It is NOT the "
                          "uncertainty of the nuclear data, which is not propagated here."),
        "disclaimer": DISCLAIMER,
    }
