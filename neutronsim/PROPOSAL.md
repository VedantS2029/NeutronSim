# Proposal (workflow step 1) — Educational Neutron–Nucleus Reaction Model

**EDUCATIONAL MODEL — not a tool for laboratory, reactor, or weapons design.**

## Data and processing
| Item | Choice |
|---|---|
| Evaluated library | ENDF/B-VIII.0 (Brown et al., Nucl. Data Sheets 148, 1 (2018), doi:10.1016/j.nds.2018.02.001). Version is read from each file's own header flags (NLIB, NVER, LREL). |
| File origin | Copies in NJOY2016 `tests/resources` (commit `ab27c64…`), SHA-256 pinned. NNDC/IAEA were unreachable, so bytes were **not** compared with the official distribution. |
| Resonance region | Raw MF3 is incomplete there. NJOY2016 (2016.80) **RECONR**, err = 0.001, T = 0 K, is run by `scripts/build_data.py`. |
| Not applied | Doppler broadening, URR self-shielding/probability tables, S(α,β). |

## Supported scope (everything else: "unsupported by available data")
| Isotope | Energy range (eV) | Channels with data |
|---|---|---|
| B-10 | 1e-5 – 2e7 | elastic, inelastic, (n,γ), (n,p), (n,d), (n,t), (n,α), (n,t+2α) |
| Al-27 | 1e-5 – 1.5e8 | elastic, inelastic, (n,γ), (n,2n), several (n,charged), (n,anything) |
| Ag-109 | 1e-5 – 2e7 | elastic, inelastic, (n,γ), (n,2n), (n,3n), (n,p), (n,α), others |
| U-235 | 1e-5 – 3e7 | elastic, inelastic, (n,γ), **(n,f)**, (n,2n), (n,3n), (n,anything) |

Fission is reported as a probability channel only. Radioactive decay and fusion are separate topics and are not modelled.

## Calculations (each labelled DATA / CALCULATION / ASSUMPTION in output)
1. DATA: σ_i(E) by interpolation exactly as stored; never outside stored range; threshold tables treated as zero below threshold (first value exactly 0).
2. CALCULATION: P_i = σ_i/σ_total, conditional on an interaction. Summation check total − Σ channels is shown; no renormalisation hidden.
3. CALCULATION: multinomial sampling of outcomes, with Monte Carlo error reported separately from data uncertainty.

## Known limitations
- Uncertainties **not propagated** (covariances, MF33, exist in the source for some channels; reader not implemented).
- Reaction products (residual nuclides, emitted-particle spectra) not read → unsupported.
- No rates, mean free paths, materials, geometry, multiplicity, chain behaviour, or criticality — excluded by design.
- Reference values are from a 1992 hand-entered compilation (see README); one case (U-235) is flagged DISCREPANT.
