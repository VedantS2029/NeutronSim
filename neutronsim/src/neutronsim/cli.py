"""Command line interface. Every printed number is tagged with a source record or a calculation."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from . import core, notes, validation
from .nuclide import NuclideError

BANNER = "=" * 78 + "\n  EDUCATIONAL MODEL - not a tool for laboratory, reactor, or weapons design.\n" + "=" * 78


def fmt(x: float | None) -> str:
    return "unsupported" if x is None else f"{x:.6g}"


def _source_line(tag: str, s: dict) -> str:
    lo, hi = s["energy_range_eV"]
    return (f"[{tag}] {s['dataset']} | {s['isotope']} (MAT {s['mat']}) | {s['file']} {s['channel']} | "
            f"stored range {lo:g}-{hi:g} eV | {s['units']} | interpolation {s['interpolation'] or 'n/a'}")


def _common_source(s: dict) -> str:
    return (f"all of the above: source file {s['source_file']} (sha256 {s['source_sha256']})\n"
            f"     -> processed file {s['processed_file']} (sha256 {s['processed_sha256'][:16]}...)\n"
            f"     processing: {'; '.join(s['processing'])}")


def render_xs(ds: core.Dataset, res: core.Result) -> str:
    out = [BANNER, f"Target: {res.isotope}    Incident neutron energy: {res.energy_eV:.6g} eV"]
    lo, hi = ds.energy_range_eV
    out.append(f"Dataset: {ds.library}, evaluated range {lo:g}-{hi:g} eV")
    if not res.supported:
        out.append("\nRESULT: " + res.reason)
        return "\n".join(out)
    sources: list[str] = []

    def tag(c: core.ChannelResult) -> str:
        sources.append(_source_line(f"S{len(sources) + 1}", c.source))
        return f"S{len(sources)}"

    out.append("\nDATA (evaluated cross sections, barn)")
    out.append(f"  {'MT':<5}{'channel':<44}{'sigma (b)':>14}  status / source")
    t = res.total
    out.append(f"  {t.mt:<5}{t.name:<44}{fmt(t.sigma_barn):>14}  {tag(t)}")
    for c in res.channels:
        label = c.status if c.status != "ok" else tag(c)
        out.append(f"  {c.mt:<5}{c.name:<44}{fmt(c.sigma_barn):>14}  {label}")
    notes_ = [c for c in res.channels if c.note]
    if notes_:
        out.append("  notes:")
        for c in notes_:
            out.append(f"    MT{c.mt}: {c.note}")

    out.append("\nCALCULATION: probability that an interaction is of each type = sigma_i / sigma_total")
    for c in sorted(res.channels, key=lambda c: -(c.probability or 0)):
        if c.probability:
            out.append(f"  MT{c.mt:<4}{c.name:<44}{c.probability:>12.6g}   [{c.kind}]")
    out.append(f"  summation check: total - sum(listed channels) = {res.remainder_barn:.3g} b "
               f"({res.remainder_fraction:+.2e} of total; tolerance {core.REMAINDER_REL_TOL:g})"
               + (f"\n  FLAG: {res.remainder_flag}" if res.remainder_flag else ""))

    out.append("\nUNCERTAINTY: not propagated in this version. Per channel:")
    seen = set()
    for c in [t] + res.channels:
        if c.uncertainty not in seen:
            seen.add(c.uncertainty)
            out.append(f"  - {c.uncertainty}")
    out.append("  Digits above are numerical precision, not physical certainty.")
    out.append("\nSOURCES (DATA)")
    out.extend("  " + s for s in sources)
    out.append("  " + _common_source(t.source))
    out.append("\nASSUMPTIONS (model, not data)")
    out.extend(f"  - {a}" for a in res.assumptions)
    out.append("  - Radioactive decay: unsupported by available data (not a neutron-induced channel).")
    out.append("  - Reaction products (residual nuclides, emitted-particle energies/angles): unsupported by "
               "available data (product files not read). Only the channel label is reported.")
    return "\n".join(out)


def render_sample(res: core.Result, s: dict) -> str:
    out = [BANNER, f"Sampling {s['n']} outcomes for {res.isotope} at {res.energy_eV:.6g} eV (seed {s['seed']})",
           "These are random draws from the model probabilities below. They do not predict any real event.",
           f"\n  {'outcome':<52}{'p (model)':>11}{'count':>9}{'observed':>10}{'+/- MC':>10}"]
    for r in s["rows"]:
        out.append(f"  {r['outcome']:<52}{r['model_probability']:>11.4g}{r['count']:>9}"
                   f"{r['observed_fraction']:>10.4g}{r['sampling_std_error']:>10.2g}")
    out.append(f"\n  {s['error_meaning']}")
    out.append(f"  normalisation factor {s['normalisation_factor']:.8g}: {s['normalisation_note']}")
    return "\n".join(out)


def cmd_list(_a) -> int:
    import json as _j
    man = _j.loads((core.DATA_DIR / "MANIFEST.json").read_text())
    print(BANNER)
    print(f"Data source: {man['source']['description']}\n  {man['source']['repository']} @ {man['source']['commit'][:10]}")
    print(f"  caveat: {man['source']['caveat']}")
    print(f"Processing: {man['processing']['code']}, RECONR {man['processing']['reconr']}")
    print("  not applied: " + "; ".join(man['processing']['not_applied']))
    for iso in sorted(man["isotopes"]):
        ds = core.Dataset(iso)
        lo, hi = ds.energy_range_eV
        print(f"\n{iso}: {ds.library}, {lo:g}-{hi:g} eV")
        print("  channels: " + ", ".join(f"MT{mt}" for mt in [2] + ds.partial_mts))
    print("\nEverything else: unsupported by available data (no universal isotope coverage).")
    return 0


def _need_data(isotope: str, unit_hint: str) -> str:
    """Normalise the isotope; if it has no evaluated data, point to the model-estimate commands instead of just failing."""
    iso = core.normalize_isotope(isotope)
    from . import explorer
    if iso not in explorer.data_isotopes():
        raise core.UnsupportedError(
            f"{iso}: unsupported by available data (no evaluated data for it). For a physics-law MODEL ESTIMATE with "
            f"random draws run: neutronsim simulate {iso} --energy <E>   (or 'forecast' for cross sections only).")
    return iso


def cmd_xs(a) -> int:
    ds = core.Dataset(_need_data(a.isotope, "xs"))
    res = core.evaluate(ds, core.to_eV(a.energy, a.unit))
    print(json.dumps(asdict(res), indent=2) if a.json else render_xs(ds, res))
    return 0 if res.supported else 3


def cmd_sample(a) -> int:
    ds = core.Dataset(_need_data(a.isotope, "sample"))
    res = core.evaluate(ds, core.to_eV(a.energy, a.unit))
    s = core.sample_outcomes(res, a.n, a.seed)
    print(json.dumps(s, indent=2) if a.json else render_sample(res, s))
    return 0


def cmd_validate(a) -> int:
    rs = validation.run()
    print(BANNER)
    print(f"Reference: Sears (1992) thermal absorption, E = {validation.e_2200_eV():.7g} eV (2200 m/s). "
          "Tolerance k=3 on the reference uncertainty only; evaluation uncertainty not propagated.")
    print(f"\n  {'case':<24}{'evaluated':>11}{'reference':>16}{'diff %':>8}{'sigma':>7}  status")
    bad = 0
    for r in rs:
        print(f"  {r.id:<24}{r.evaluated_b:>11.5g}{r.reference_b:>11.5g}+/-{r.reference_u_b:<5.3g}{100 * r.diff_rel:>8.2f}"
              f"{r.sigma_units:>7.2f}  {r.status}")
        if r.note:
            print(f"      note: {r.note}")
        bad += r.status != r.expected_status
    print("\nDISCREPANT = outside k*u_ref but inside the 2% gross-error bound; reported, not hidden.")
    return 1 if bad else 0


def cmd_explain(a) -> int:
    print(BANNER)
    keys = [a.topic] if a.topic else list(notes.TOPICS)
    for k in keys:
        print(f"\n[{k}] {notes.TOPICS[k]}")
    return 0


def cmd_gui(_a) -> int:
    from . import gui
    return gui.main()


def cmd_predict(a) -> int:
    from . import chem, explorer, physics
    from .nuclide import Nuclide
    nuc = Nuclide.parse(a.nuclide)
    print(BANNER)
    print(nuc.describe())
    print("\n".join(physics.render(physics.estimate_nuclide(nuc))))
    print("\n".join(chem.render(nuc)))
    print("  Measured neutron data in this project: " + ("YES (see 'xs')" if nuc.key in explorer.data_isotopes() else "no"))
    return 0


def cmd_compare(a) -> int:
    from . import compare, physics
    from .nuclide import Nuclide
    nuc = Nuclide.parse(a.nuclide)
    print(BANNER)
    print(nuc.describe())
    print("\n".join(physics.render(physics.estimate_nuclide(nuc))))
    print()
    print("\n".join(compare.lines(nuc, core.to_eV(a.energy, a.unit), full=True)))
    return 0


def cmd_forecast(a) -> int:
    from . import chem, nphys, physics
    from .nuclide import Nuclide
    nuc = Nuclide.parse(a.nuclide)
    print(BANNER)
    print(nuc.describe())
    print("\n".join(nphys.render(nphys.forecast(nuc, core.to_eV(a.energy, a.unit)))))
    if nuc.key in __import__("neutronsim.explorer", fromlist=["x"]).data_isotopes():
        print(f"  Real measured data exists for {nuc.key}: run 'neutronsim compare {nuc.key} --energy {a.energy:g} --unit {a.unit}'.")
    return 0


def cmd_simulate(a) -> int:
    """Random-draw simulation for ANY nuclide from physics laws (model estimate; refused only where the compound nucleus is too light or too weakly bound)."""
    from . import estsim, explorer
    from .nuclide import Nuclide
    nuc = Nuclide.parse(a.nuclide)
    fc, res, s = estsim.simulate(nuc, core.to_eV(a.energy, a.unit), a.n, a.seed)
    if a.json:
        print(json.dumps({"model_estimate": asdict(fc), "sample": s}, indent=2, default=str))
        return 0
    print(BANNER)
    print(nuc.describe())
    print(estsim.render_simulation(fc, res, s))
    if nuc.key in explorer.data_isotopes():
        print(f"\n  Real evaluated data exists for {nuc.key}: 'neutronsim sample {nuc.key} --energy {a.energy:g} --unit {a.unit}' uses it.")
    return 0


def cmd_box(a) -> int:
    """Neutrons released inside a block of one (non-fissile) isotope: how many are left, how much heat is deposited."""
    from . import box
    r = box.run_box(a.isotope, core.to_eV(a.energy, a.unit), density_g_cm3=a.density, density_factor=a.density_factor,
                    side_cm=a.size, n=a.n, seed=a.seed, t_end_s=a.t_end, source=a.source)
    if a.json:
        d = asdict(r)
        print(json.dumps(d, indent=2))
        return 0
    print(BANNER)
    print(box.render(r))
    return 0


def cmd_boxgui(a) -> int:
    from . import boxgui
    return boxgui.main(a.isotope)


def cmd_boxviz(a) -> int:
    from . import boxviz
    return boxviz.main(a.isotope)


def cmd_all(a) -> int:
    from . import runall
    return runall.main(verbose=a.verbose, run_tests=not a.no_tests, gui=a.gui)


def cmd_forecastcheck(_a) -> int:
    from . import nphys
    print("\n".join(nphys.check()[0]))
    return 0


def cmd_modelcheck(_a) -> int:
    from . import chem, physics
    lines, _stats = physics.check()
    print("\n".join(lines))
    print()
    print("\n".join(chem.check()[0]))
    return 0


def cmd_explore(a) -> int:
    from . import explorer
    return explorer.main(a.isotope)


def cmd_viz(a) -> int:
    from . import estsim, explorer, viz
    from .nuclide import Nuclide
    iso = core.normalize_isotope(a.isotope)
    e_eV = core.to_eV(a.energy, a.unit)
    if iso not in explorer.data_isotopes():        # no evaluated data: animate the labelled model estimate instead
        res = estsim.visual_result(Nuclide.parse(iso), e_eV)
        print(f"NO EVALUATED DATA for {iso}: animating the physics-law MODEL ESTIMATE." + (f"\n{res.reason}" if not res.channels else ""))
        return viz.main(res)
    ds = core.Dataset(iso)
    res = core.evaluate(ds, e_eV)
    if not res.supported:
        print(res.reason, file=sys.stderr)
        return 3
    return viz.main(res)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neutronsim", description="Educational neutron-nucleus reaction model "
                                "(evaluated data, probabilistic, traceable). " + core.DISCLAIMER)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="supported isotopes, channels, energy ranges").set_defaults(fn=cmd_list)
    for name, fn, h in (("xs", cmd_xs, "channel cross sections and probabilities"),
                        ("sample", cmd_sample, "random draws from channel probabilities")):
        s = sub.add_parser(name, help=h)
        s.add_argument("isotope", help="e.g. U-235, b10, 27Al")
        s.add_argument("--energy", type=float, required=True, help="incident neutron energy (required)")
        s.add_argument("--unit", default="eV", choices=list(core.UNITS))
        s.add_argument("--json", action="store_true")
        if name == "sample":
            s.add_argument("-n", type=int, default=1000)
            s.add_argument("--seed", type=int, default=None)
        s.set_defaults(fn=fn)
    sub.add_parser("gui", help="open the step-by-step setup wizard").set_defaults(fn=cmd_gui)
    x = sub.add_parser("explore", help="periodic-table isotope explorer (pick an element, adjust protons/neutrons/electrons)")
    x.add_argument("isotope", nargs="?", help="optional starting nuclide, e.g. U-235 or 'O-16 2-'")
    x.set_defaults(fn=cmd_explore)
    pr = sub.add_parser("predict", help="rough physics estimate for ANY nuclide: binding energy and allowed decay channels")
    pr.add_argument("nuclide", help="e.g. U-238, tritium, 14C")
    pr.set_defaults(fn=cmd_predict)
    cp = sub.add_parser("compare", help="model estimate next to real evaluated data / reference values, for any nuclide")
    cp.add_argument("nuclide", help="e.g. U-235, B-10, C-14")
    cp.add_argument("--energy", type=float, default=0.0253, help="incident neutron energy (default thermal 0.0253)")
    cp.add_argument("--unit", default="eV", choices=list(core.UNITS))
    cp.set_defaults(fn=cmd_compare)
    fp = sub.add_parser("forecast", help="what a neutron would do to ANY isotope, from physics laws (with measured error ranges)")
    fp.add_argument("nuclide", help="e.g. Na-23, Pb-208, 'Xe-136'")
    fp.add_argument("--energy", type=float, default=0.0253, help="neutron energy (default thermal 0.0253 eV)")
    fp.add_argument("--unit", default="eV", choices=list(core.UNITS))
    fp.set_defaults(fn=cmd_forecast)
    sm = sub.add_parser("simulate", help="random-draw simulation for ANY nuclide from physics laws (labelled model estimate)")
    sm.add_argument("nuclide", help="e.g. Na-23, Pb-208, C-14")
    sm.add_argument("--energy", type=float, default=0.0253, help="neutron energy (default thermal 0.0253 eV)")
    sm.add_argument("--unit", default="eV", choices=list(core.UNITS))
    sm.add_argument("-n", type=int, default=10000, help="number of random draws")
    sm.add_argument("--seed", type=int, default=None)
    sm.add_argument("--json", action="store_true")
    sm.set_defaults(fn=cmd_simulate)
    bx = sub.add_parser("box", help="neutrons released inside a block of one isotope (any isotope except fissile/actinide ones): how many are left, heat deposited")
    bx.add_argument("isotope", help="any isotope, e.g. B-10, Fe-56, Pb-208; B-10/Al-27/Ag-109 use evaluated data, the rest the labelled model estimate (fissile and actinide isotopes are refused)")
    bx.add_argument("--energy", type=float, required=True, help="starting neutron energy (required)")
    bx.add_argument("--unit", default="eV", choices=list(core.UNITS))
    dg = bx.add_mutually_exclusive_group()
    dg.add_argument("--density-factor", type=float, default=None, help="density as a multiple of the solid (default 1.0, max 2.0)")
    dg.add_argument("--density", type=float, default=None, help="density in g/cm3")
    bx.add_argument("--size", type=float, default=10.0, help="cube side in cm (default 10)")
    bx.add_argument("-n", type=int, default=1000, help="neutrons released (default 1000)")
    bx.add_argument("--seed", type=int, default=None)
    bx.add_argument("--t-end", type=float, default=0.1, help="stop the clock at this time in seconds (default 0.1)")
    bx.add_argument("--source", default="center", choices=["center", "uniform"])
    bx.add_argument("--json", action="store_true")
    bx.set_defaults(fn=cmd_box)
    bg = sub.add_parser("boxgui", help="box test as a window: density slider, live statistics and plots (needs tkinter)")
    bg.add_argument("isotope", nargs="?", help="optional starting isotope (any except fissile/actinide ones)")
    bg.set_defaults(fn=cmd_boxgui)
    bv = sub.add_parser("boxviz", help="box test window opened on the Watch tab: neutron paths animated inside the block (needs tkinter)")
    bv.add_argument("isotope", nargs="?", help="optional starting isotope (any except fissile/actinide ones)")
    bv.set_defaults(fn=cmd_boxviz)
    al = sub.add_parser("all", help="run everything (checks, examples, simulations, test suite) and print a pass/fail summary")
    al.add_argument("-v", "--verbose", action="store_true", help="show the full output of every step")
    al.add_argument("--no-tests", action="store_true", help="skip the pytest suite")
    al.add_argument("--gui", action="store_true", help="after everything passes, open the GUI (wizard, with explorer and visualizer)")
    al.set_defaults(fn=cmd_all)
    sub.add_parser("forecastcheck", help="how far the forecast is from real data").set_defaults(fn=cmd_forecastcheck)
    sub.add_parser("modelcheck", help="how far the physics model is from reference values").set_defaults(fn=cmd_modelcheck)
    v = sub.add_parser("viz", help="animated visualizer: fire neutrons at a nucleus of protons and neutrons")
    v.add_argument("isotope", help="e.g. U-235, b10, 27Al")
    v.add_argument("--energy", type=float, required=True, help="incident neutron energy (required)")
    v.add_argument("--unit", default="eV", choices=list(core.UNITS))
    v.set_defaults(fn=cmd_viz)
    sub.add_parser("validate", help="compare against published reference values").set_defaults(fn=cmd_validate)
    e = sub.add_parser("explain", help="qualitative notes")
    e.add_argument("topic", nargs="?", choices=list(notes.TOPICS))
    e.set_defaults(fn=cmd_explain)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (core.InputError, core.UnsupportedError, NuclideError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except core.DataIntegrityError as exc:
        print(f"data integrity error: {exc}", file=sys.stderr)
        return 4
