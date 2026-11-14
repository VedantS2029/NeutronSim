# Passover checklist — Educational Neutron–Nucleus Simulator

- [x] 1. Check data availability. NNDC/IAEA blocked. Real ENDF/B-VIII.0 files found in NJOY2016 tests/resources (commit ab27c64). NJOY2016 built from source for RECONR (resonance reconstruction needed; raw MF3 alone is wrong in resonance range).
- [x] 2. PROPOSAL.md written
- [x] 3. Fetched+verified (sha256) B-10, Al-27, Ag-109, U-235 ENDF/B-VIII.0; NJOY2016 RECONR (0 K, err 0.001) -> data/pendf; data/MANIFEST.json
- [x] 4. Reference values: Sears 1992 via NIST NCNR (B-10, Al-27, Ag-109, U-235) -> validation/reference_cases.json
- [x] 5. Loader: src/neutronsim/endf.py + channels.py (written)
- [x] 6. Calc layer: core.py (provenance, thresholds, no extrapolation, probabilities, sampling)
- [x] 7. CLI: cli.py (xs, sample, validate, explain, list; input validation)
- [x] 8. Tests: 82 tests incl. independent endf-package cross-parse, tamper check, scope exclusions
- [x] 9. validation/sample_run.txt (validate, xs, sample, out-of-range refusal)
- [x] 10. README.md
- [x] 11. Packaged + presented


# Isotope explorer (passover steps)
- [x] E1. nuclide.py - particle model (protons/neutrons/electrons), parsing, periodic-table positions, tests
- [x] E2. explorer.py - periodic-table picker, +/- controls, `neutronsim explore`, tests (window checked only against the stub Tk)
- [x] E3. physics.py: semi-empirical mass formula, Q-values per channel, verdict (stable/borderline/unstable/particle-unbound), `predict` + `modelcheck`, tests; accuracy measured against validation/physics_reference.json
- [x] E4. compare.py: model vs ENDF Q-values + measured reaction shares + Sears check for the 4 loaded isotopes; reference values for a few others; explicit "no data" otherwise; `compare` CLI, explorer energy drop-down, tests
- [x] E5. chem.py (configurations, masses, H-like binding); explorer opens from the wizard's isotope page (ExplorerWindow, data-gated); multi-digit charges parse (U-238 91+); `predict`/`modelcheck` include chemistry; tests; docs


# Forecast for any isotope (passover steps, started after E5)
- [x] F6. nphys.py: kinematics, potential scattering, black-disk fast cross sections, statistical capture (Fermi gas + Brink-Axel + average resonance),
      calibration from 34 thermal capture values, regimes with measured error ranges, `forecast` / `forecastcheck`, explorer integration, tests
- [ ] F7. statistical-model channel competition (Weisskopf-Ewing / Hauser-Feshbach-lite): (n,n'), (n,2n), (n,p), (n,alpha) from compound-nucleus decay;
      transmission coefficients; fills the 1 keV - 5 MeV gap; validate against ENDF for the 4 isotopes
- [ ] F8. "test runner": energy sweeps for any isotope, random-draw outcomes, product chains (what the product does next), CSV export
- [ ] F9. GUI/wizard integration of the forecast and sweeps, final validation report, docs


# Box test: neutrons inside a block of one isotope (passover steps; added after F6)
Scope: non-fissile isotopes only (B-10, Al-27, Ag-109). Fission-capable data (U-235) is refused by design: a fissile
block with a density knob and a heat counter would be a chain-reaction / criticality calculation (see README, Limitations).
- [x] B1. box.py engine (Monte Carlo in a cube, vacuum outside, evaluated cross sections, density in g/cm3 or x solid, stats:
      absorbed / leaked / still alive, collisions by type, heat in MeV and J, energy bookkeeping check, time series), `neutronsim box` CLI,
      tests/test_box.py (26 tests: bookkeeping, B-10 heat = 2.79 MeV per absorption, density effect, refusals, validation)
- [x] B2. boxgui.py: density slider (log scale), box size, energy, neutrons, live stats + population/heat-vs-time plots; `boxgui` CLI,
      run_gui.py boxgui, wizard Tools menu + Results button; tests/test_boxgui.py (model + stub-Tk window + wizard hand-off). Window checked against the stub Tk only, not on a real display
- [x] B3. docs: README section, `explain box` note, run_all.py steps (incl. the U-235 refusal), full test run


# Multi-atom (box) visualizer: make it work + as accurate as possible (passover steps)
- [x] V1. Baseline: install, run full test suite, run boxviz headless / real Tk under Xvfb, note failures
- [x] V2. Audit box.py physics (kinematics, sampling, bookkeeping, timing) against independent checks (analytic attenuation, slab/cube escape, elastic energy loss, closure)
- [x] V3. Audit boxviz.py (what is drawn vs what was simulated); list every schematic element
- [x] V4. Fix bugs / replace schematic parts with real simulated data where possible
- [x] V5. Add tests for each fix + analytic validation tests
- [x] V6. Re-run full suite + GUI under Xvfb, screenshot
- [x] V7. Update README / PLAN, package zip, report
