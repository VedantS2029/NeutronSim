"""Simulation for nuclides WITHOUT evaluated data, driven by the physics-law forecast of nphys.

The forecast gives cross sections from nuclear-physics laws (geometric size, potential scattering, compound-nucleus
statistics). This module turns them into interaction PROBABILITIES (sigma_i / sigma_total, as for real data) and
reuses core.sample_outcomes and the visualizer on them, so a model estimate behaves like a data set everywhere
while staying labelled as a model.

A set of probabilities is only built where the forecast covers EVERY open channel:
  * below 1 keV:   elastic + radiative capture (the forecast states no other channel is open)
  * 1 keV - 20 MeV: the channels of hf.py (optical model + Hauser-Feshbach-type decay): elastic MT2, inelastic MT4, (n,2n) MT16,
    (n,3n) MT17, (n,p) MT103, (n,alpha) MT107, capture MT102, and the multi-particle channels the decay tree produces
    (MT22, 24, 28, 41, 111, 112). Each has an MT number the visualizer already knows.
  * from 5 MeV where hf.py has no channels (above 20 MeV, or the nucleus is too light for it): elastic + non-elastic
    (geometric estimate, as before; which reaction is not predicted)
  * capture not predictable (too light, too weakly bound): refused, because "100 % elastic" would be false.
    Between 1 keV and 5 MeV the same condition leaves no channel set, so it is refused there.

Two kinds of error are kept apart. Monte Carlo noise shrinks with more draws. Model uncertainty does not: it is the
measured scatter of the forecast against real data (nphys.check), shown as a probability range per outcome.
"""
from __future__ import annotations

from . import core, hf as hf_mod, nphys
from .nuclide import Nuclide

MODEL_ASSUMPTIONS = [
    "MODEL ESTIMATE: cross sections come from physics laws (nphys), not from an evaluated data library.",
    "Probabilities are conditional on an interaction having occurred, as for evaluated data.",
    "The probability ranges are the forecast's measured scatter against known cases, not guarantees.",
    "Isolated target nucleus at rest; no resonance structure, shell effects, fission, pre-equilibrium or direct inelastic scattering.",
    "From 1 keV: optical-model transmission + Weisskopf-Ewing (spin-averaged) Hauser-Feshbach decay; discrete levels are estimated.",
]
E_LOW_MAX_eV = nphys.E_LOW_MAX


def _row(fc: nphys.Forecast, prefix: str) -> nphys.Row | None:
    return next((r for r in fc.rows if r.name.startswith(prefix)), None)


def _parts(fc: nphys.Forecast) -> list[tuple[int, str, nphys.Row]]:
    """(MT, label, forecast row) for each outcome, or raise UnsupportedError saying why none can be built."""
    if fc.energy_eV <= E_LOW_MAX_eV:
        el, cap = _row(fc, "elastic"), _row(fc, "capture")
        if cap is None or cap.value_b is None:
            raise core.UnsupportedError(
                f"cannot simulate {fc.key}: the model cannot predict its capture ({cap.note if cap else 'no capture row'}), "
                "so the outcome probabilities would be wrong.")
        return [(2, "elastic scattering", el), (102, "radiative capture", cap)]
    if fc.channels:
        parts = [(mt, _label(mt), row) for mt, row in sorted(fc.channels.items()) if row.value_b and row.value_b > 0]
        if not parts:
            raise core.UnsupportedError(f"cannot simulate {fc.key}: the model predicts no open channel at this energy.")
        return parts
    if fc.energy_eV >= nphys.E_FAST_MIN:      # no hf.py channels (above 20 MeV, or nucleus too light): geometric split
        el, non = _row(fc, "elastic"), _row(fc, "nonelastic")
        return [(2, "elastic (shadow) scattering", el), (3, "nonelastic (all reactions together)", non)]
    raise core.UnsupportedError(
        f"cannot simulate {fc.key} between 1 keV and 5 MeV: the compound nucleus is too light or too weakly bound for the "
        "optical-model + Hauser-Feshbach model, so the split into outcomes would be invented. "
        "Use a nuclide with evaluated data, or an energy below 1 keV where capture is predicted.")


def _label(mt: int) -> str:
    return {2: "elastic scattering", 102: "radiative capture", 4: "inelastic scattering (n,n')"}.get(mt) or hf_mod.LABEL_OF.get(mt, f"MT{mt}")


def can_simulate(nuc: Nuclide, e_eV: float) -> str | None:
    """None if outcome probabilities can be built for this nuclide and energy, else the reason why not."""
    try:
        _parts(nphys.forecast(nuc, e_eV))
    except core.UnsupportedError as exc:
        return str(exc)
    return None


def estimate_result(fc: nphys.Forecast) -> core.Result:
    """The forecast as a core.Result (probabilities per channel), usable by core.sample_outcomes and the visualizer."""
    parts = _parts(fc)
    total = sum(r.value_b for _, _, r in parts)
    chans = []
    for mt, label, r in parts:
        lo, hi = _prob_range(parts, r)
        chans.append(core.ChannelResult(
            mt, label, "model estimate", "ok", r.value_b, probability=r.value_b / total,
            note=f"{r.note}; p in {lo:.3g}-{hi:.3g}", uncertainty=f"model scatter factor x/÷{r.factor:g}", source=None))
    tot = core.ChannelResult(1, "(n,total) model estimate", "model estimate", "ok", total,
                             uncertainty="model scatter, see per-channel factors", source=None)
    res = core.Result(fc.key, fc.energy_eV, supported=True, total=tot, channels=chans, remainder_barn=0.0,
                      remainder_fraction=0.0, assumptions=list(MODEL_ASSUMPTIONS))
    res.disclaimer = core.DISCLAIMER + " MODEL ESTIMATE - no evaluated data."
    return res


def visual_result(nuc: Nuclide, e_eV: float) -> core.Result:
    """A Result the visualizer can open for ANY nuclide. Where outcome probabilities exist they are the model's; where they do
    not (capture unpredictable) the table is empty: the nucleus is drawn but no neutron can be fired."""
    fc = nphys.forecast(nuc, e_eV)
    try:
        return estimate_result(fc)
    except core.UnsupportedError as exc:
        res = core.Result(fc.key, e_eV, supported=True, remainder_barn=0.0, remainder_fraction=0.0,
                          assumptions=list(MODEL_ASSUMPTIONS))
        res.reason = str(exc)
        res.disclaimer = core.DISCLAIMER + " MODEL ESTIMATE - no evaluated data."
        return res


def _prob_range(parts, row: nphys.Row) -> tuple[float, float]:
    """Probability range of one outcome when every cross section moves independently by its own factor (extremes)."""
    f = row.factor or 1.0
    others = [(r.value_b, r.factor or 1.0) for _, _, r in parts if r is not row]
    lo = row.value_b / f / (row.value_b / f + sum(v * g for v, g in others))
    hi = row.value_b * f / (row.value_b * f + sum(v / g for v, g in others))
    return lo, hi


def prob_ranges(fc: nphys.Forecast) -> dict[int, tuple[float, float]]:
    parts = _parts(fc)
    return {mt: _prob_range(parts, r) for mt, _, r in parts}


def simulate(nuc: Nuclide, e_eV: float, n: int, seed: int | None = None) -> tuple[nphys.Forecast, core.Result, dict]:
    """Forecast -> probabilities -> n random draws. Raises core.InputError / core.UnsupportedError."""
    fc = nphys.forecast(nuc, e_eV)
    res = estimate_result(fc)
    s = core.sample_outcomes(res, n, seed)
    s["model_probability_range"] = {f"MT{mt}": rg for mt, rg in prob_ranges(fc).items()}
    s["model_estimate"] = True
    return fc, res, s


HEADER = ("NO EVALUATED DATA for {key} in this project - this is a MODEL ESTIMATE from physics laws, not an evaluated or "
          "measured result.")


def render_simulation(fc: nphys.Forecast, res: core.Result, s: dict) -> str:
    from .cli import BANNER, render_sample
    rng = prob_ranges(fc)
    out = [HEADER.format(key=fc.key),
           "Nothing below carries a dataset, version or evaluated uncertainty. Educational model only.", "",
           "\n".join(nphys.render(fc)), "",
           "MODEL PROBABILITIES (sigma_i / sigma_total) and how far the model could be off:"]
    for c in res.channels:
        lo, hi = rng[c.mt]
        out.append(f"  MT{c.mt:<4}{c.name:<40}p = {c.probability:<9.4g} plausible range {lo:.3g} - {hi:.3g}")
    out += ["", render_sample(res, s).replace(BANNER + "\n", "", 1), "",
            "  The +/- MC column shrinks with more draws. The plausible range above does not: it is the model's own error,",
            "  so more draws make the counts smoother, not more correct."]
    return "\n".join(out)
