# neutronsim — educational neutron–nucleus reaction model

> **EDUCATIONAL MODEL — not a tool for laboratory, reactor, or weapons design.**
> Results are probabilities. A random sample illustrates a distribution; it does not predict what any real neutron will do.

See `PROPOSAL.md` (scope, data, limitations) and `PLAN.md` (checklist).

> **Status: seeking expert review.** Most of this code was written with the help of an AI assistant (Claude) by a high-school
> student with no lab and no formal nuclear-physics training, so the physics has **not** been independently checked.
> Known weak points are listed under *Validation* and *Limitations* below (for example, the U-235 thermal-absorption
> discrepancy of +0.78 % against Sears 1992). Corrections and criticism are very welcome: please open an issue.

## Setup
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"      # stdlib-only at runtime; 'endf' + pytest for tests
pytest                        # ~500 tests, ~6 min (the U-235 file is large)
```
## Use
```bash
python -m neutronsim gui                                     # step-by-step setup wizard (needs tkinter)
python -m neutronsim explore [U-235]                         # periodic-table isotope explorer (needs tkinter)
python -m neutronsim viz B-10 --energy 0.0253 --unit eV  # animated visualizer (needs tkinter)
python -m neutronsim box Fe-56 --energy 14 --unit MeV --density-factor 0.1   # neutrons inside a block of ANY one isotope (not fissile/actinide)
python -m neutronsim boxgui [Fe-56]                          # the same as a window: density slider, live stats, plots, animation tab (needs tkinter)
python -m neutronsim boxviz [Fe-56]                          # the same window opened on the animation ("Watch the neutrons") tab
python -m neutronsim list
python -m neutronsim xs B-10 --energy 0.0253 --unit eV      # energy is required
python -m neutronsim xs U-235 --energy 1 --unit MeV --json
python -m neutronsim sample Ag-109 --energy 1 --unit keV -n 10000 --seed 1
python -m neutronsim validate
python -m neutronsim explain [probability|energy|channels|capture|scattering|fission|decay|fusion]
```
Sample output: `validation/sample_run.txt`.

## Visualizer
`neutronsim viz ISOTOPE --energy E` (or the **Visualize...** button on the wizard's Results page) shows the nucleus as
its protons (red) and neutrons (blue). **Fire neutron** draws one outcome from the model probabilities and animates it;
the tally panel shows your shots building up toward those probabilities.
By default the nucleus is NOT reset after a shot: it continues from the product (capture gives the next isotope, (n,p) the residual, scattering leaves it
as it is) and the next neutron is fired at that nucleus, with its own outcome table and tally (evaluated data where the project has it, otherwise the
labelled model estimate). Fission destroys the nucleus, so it restarts from the starting isotope. Untick **Continue from the product** for the old
behaviour (always the starting isotope); **Back to ISOTOPE** returns to the starting nucleus without clearing any tally.
Real: the outcome draw (same probabilities as `sample`) and the nucleon counts from the isotope name.
Bookkeeping, not data: residual nuclei conserve Z and A (the evaluated files contain no reaction products).
Schematic: all positions, motion, angles and colours. Fission is drawn only as "the nucleus splits"; fragments,
emitted neutrons, energy release and chain reactions are not modelled, by design.

## Box test (`box`, `boxgui`)
Neutrons are released inside a cube made of ONE isotope (vacuum outside) and followed collision by collision, using the evaluated
cross sections. Each ends absorbed, leaked out, or still alive when the clock stops. You set the isotope, neutron energy, **density**
(slider, 1e-4 to 2 x solid; or `--density` in g/cm3), box side and number of neutrons. The window shows live statistics (how many are
left / absorbed / leaked, collisions, heat in MeV and J, mean free path) and two plots against time (population, cumulative heat);
open it with `python run_gui.py boxgui`, from the wizard's **Tools > Box test...** or the **Box test...** button on the Results page.
Checks built in: every neutron is accounted for and `source energy + reaction Q = heat + energy carried out by neutrons` closes to ~1e-11 MeV;
B-10 at thermal energy gives 2.79 MeV of heat per absorbed neutron, the (n,alpha) Q-value.

**Any isotope.** Type any isotope into the box window (or pass it to `box`): B-10, Al-27 and Ag-109 run on the evaluated data; every other isotope
(Fe-56, Pb-208, Li-6, ...) runs on the same labelled **model estimate** as the single-atom visualizer (`nphys` / `hf.py` cross sections tabulated on an
energy grid, Q-values from the mass formula, reference densities typed from memory in `box.ELEMENT_DENSITY`). Every printed result and the window say
MODEL ESTIMATE. Where the model cannot split the outcomes (for example H-1, or a very light nucleus between 1 keV and 5 MeV) the box refuses and says why.
**Refused by design:** isotopes whose data contain a fission channel (U-235) and all actinides (Z >= 89): a block of fissile-capable material with a density
control and a heat counter is a chain-reaction / criticality calculation, which this model excludes; and for actinides the model does not even include
fission, so the answer would be missing the dominant process.
**Assumptions** (printed with every result): 0 K data, targets at rest, so a neutron's energy is never lowered below 0.0253 eV; elastic
scattering isotropic in the centre-of-mass frame; emitted-neutron energies in inelastic and (n,2n)-type reactions come from an evaporation
model (the data hold no spectra); gammas, charged particles and recoils stop where they are made (real capture gammas partly escape a small
block, so real heating is lower); Q-values are the evaluation's where stored, else implied by the threshold energy, else typed from memory
(two B-10 values) or the rough mass model (a per-channel source table is printed). Reference solid densities (B 2.34, Al 2.70, Ag 10.49 g/cm3)
are typed from memory.

## Watching the box (`boxviz`, the "Watch the neutrons" tab)
After each run the box window's second tab animates the **recorded histories** of the first N released neutrons (the **Watch** field, default 60) in a rotatable 3-D
cube: each neutron moves between collisions, flashes where it scatters or reacts, turns into a red cross where it is absorbed, a green dot where it leaks out.
The side panel tallies the events as they happen; the caption gives the real time and energy of the latest collision. Controls: play/pause, restart, speed, loop,
slow auto-rotation, drag to rotate.
Real: every path, collision type, energy and time is the actual history from the run (nothing is re-simulated for the picture), so the picture and the
statistics always agree. **Atoms:** the engine never places individual nuclei (solid density is ~1e22 per cm3), so the picture does not invent any. The rings
(checkbox "Struck nuclei") mark the places where a nucleus was actually hit, one per real collision, coloured by what happened; "Full paths" keeps each
neutron's whole track on screen. The side panel states the true scales of the box: atoms per cm3, atoms in the box, atomic spacing, mean free path in atomic
spacings, box size in mean free paths, spacing in nuclear diameters (nuclear radius = the 1.2 A^(1/3) fm rule of thumb). Schematic: the animation clock (each
segment takes the same time on screen; the caption gives real time and real flight length) and the 3-D projection. Not drawn: gammas, charged particles, recoils.

Engine accuracy checks (tests/test_box_accuracy.py): neutron speed is relativistically exact (2200 m/s at 0.0253 eV); direction sampling is exact at the
poles; elastic energy loss and mean scattering cosine match the textbook values; uncollided leakage from a cube centre matches the analytic average within
statistics; every neutron is accounted for and energy closes. Largest remaining physics limit: no thermal-motion (free-gas) scattering, so a thermal neutron
never gains or loses energy below 0.0253 eV (see the printed assumptions).

## Isotope explorer (in progress, built step by step)
`neutronsim explore` opens a periodic table (all 118 elements). Click an element, then use the +/- buttons:
**protons** change the element, **neutrons** change the isotope, **electrons** change only the ion charge (never the isotope).
You can also type `U-235`, `235U`, `tritium`, `O-16 2-`. A green dot marks elements with real evaluated data in this project.
Starting isotopes (`TYPICAL_A` in `explorer.py`) are convenience defaults typed from reference memory, not read from a nuclide database.
All five build steps are done (particle model, picker, physics estimate, comparison with real data, chemistry + wizard integration).
In the setup wizard, the "Target isotope" page has an **Explore any isotope (periodic table)...** button. It opens this explorer; the
"Use this isotope in the wizard" button there is enabled only for isotopes that have real evaluated data (currently 4), so the wizard
itself still never runs on a made-up isotope.

`neutronsim predict` and the explorer also show a **chemistry/atomic** block: electron configuration (Madelung filling order, an explicit
table for ~20 neutral atoms that differ such as Cr, Cu, Ag, Au, U; cations lose their outermost electrons first, so Fe3+ is 3d5; elements 104+
are predictions), the isotope mass in u (about the molar mass in g/mol of a pure sample; the model is within 0.009 u on 11 known masses),
the ion mass (atom mass minus charge x electron mass), and for one-electron ions the Bohr and Dirac ground-state binding energies.
Ionisation energies of many-electron atoms are NOT tabulated.

## Forecast for ANY isotope (`neutronsim forecast`, new in step 6)
`neutronsim forecast Na-23 [--energy 14 --unit MeV]` predicts what a neutron would do to any nuclide, from physics laws, with an error range
that was **measured** against real data (`neutronsim forecastcheck` prints the whole comparison):
- low energy (<= 1 keV): potential scattering 4 pi R^2 (R = 1.35 A^(1/3) fm; real/model within 0.5-1.5x on 4 isotopes) and radiative capture from compound-nucleus
  statistics (Fermi-gas level density with a = A/10, Brink-Axel/TRK radiative strength, average-resonance theory);
- 1 keV - 20 MeV (`src/neutronsim/hf.py`, new): optical model (Woods-Saxon, Koning-Delaroche-type global parameters, spin-orbit omitted)
  gives the compound-nucleus, shape-elastic and total cross sections; the compound nucleus then decays statistically (Weisskopf-Ewing limit of
  Hauser-Feshbach: neutron, proton, alpha and gamma compete, repeatedly, so (n,n'), (n,2n), (n,3n), (n,p), (n,alpha), (n,np), (n,n alpha)... all appear).
  Discrete low-lying levels are **estimated** by nucleus type. Not modelled: angular-momentum/parity coupling, width fluctuations, pre-equilibrium and
  direct inelastic scattering, fission. Checked against ENDF for 4 isotopes only (in-sample); each claimed factor is the worst observed miss
  (typically x/÷3-6, capture x/÷30, charged particles x/÷8, A < 20 x/÷30). Known failure: with estimated levels, inelastic scattering just above the real
  first excited level can be missing (Na-23 at 1 MeV);
- above 20 MeV, or where the compound nucleus is too light/weakly bound for hf.py: the old geometric black-disk estimate (>= 5 MeV) or potential
  scattering (below 5 MeV, low confidence);
- afterwards: the product nucleus after capture and its model decay verdict.
Honest limits: thermal capture of one isotope is set by the nearest resonance, which is effectively random, so the forecast is a range
(x/÷13); one global calibration constant (about 24) fitted on 25 known thermal capture values lifts the raw statistical estimate (printed beside it);
heavy nuclei stay overpredicted (trend +0.43 dex per dex of A); closed-shell nuclides (Z or N = 2, 8, 20, 28, 50, 82, 126) are flagged because the model
has no shell effects (Pb-208 overpredicted ~10^5). The "inside the range" statistic is in-sample. Fission cross sections, neutron yields and chain
reactions are not modelled, by design. The explorer shows the forecast for any nuclide that has no real data.

**What was typed from memory rather than read from a database** (no network was available when this was built; check before citing):
the starting isotopes `TYPICAL_A` (explorer.py), everything in `validation/physics_reference.json` (binding energies, decay Q-values,
isotope masses, stable-nuclide list), the exception table for electron configurations (chem.py), and the light-nucleus binding energies
(d, t, He-3, He-4) in compare.py/physics.py, the 34 thermal capture cross sections and two level spacings (U-238, Au-197) in
validation/physics_reference.json / nphys.py. The ENDF data and its Q-values are the project's real data.

`neutronsim predict U-238` (works for ANY nuclide, no tkinter needed) prints a **model estimate**: binding energy from the
semi-empirical mass formula and the energy release Q of each decay channel (alpha, beta-, beta+, EC, n/p emission).
Q > 0 means *energetically allowed*, not "will happen soon": half-lives, spontaneous fission, gamma emission and shell effects are not modelled.
`neutronsim compare U-235 [--energy 14 --unit MeV]` puts that estimate next to real data. For the 4 isotopes with evaluated data it shows
the measured neutron reaction shares at the chosen energy, the thermal-absorption check against Sears (1992), and the model's reaction Q-values
next to the Q-values stored in the ENDF files (model vs data, per channel; fission and inelastic scattering are not modelled; a stored Q of 0
means "not given" and is skipped, with the model's own value listed separately as MODEL ONLY). On the 4 isotopes the model's reaction Q-values
differ from the evaluated ones by 0.3-1.2 MeV on average per isotope (worst single channel 2.6 MeV, Al-27 (n,d+alpha)).
For nuclides with only a reference value (e.g. C-14, Pb-208) it shows that; for anything else it says "no measured data or reference values loaded".
In the explorer window a drop-down picks the neutron energy (thermal, 1 keV, 1 MeV, 14 MeV) for the measured-data block.
`neutronsim modelcheck` measures how far the model is from 26 reference values typed from memory (`validation/physics_reference.json`,
~AME2020, **not read from a database**): binding energies are off by up to 8 MeV (about 1% for medium/heavy nuclei, 22% for He-4);
decay energies by 1.35 MeV on average and up to 2.7 MeV, and the model gets the allowed/forbidden answer wrong for 4 of 17 real decays
(H-3, C-14, K-40, Sr-90, all low-energy). Such cases come out as BORDERLINE rather than a guess. Known miss: the model's most-stable isobar
at A=120 is Te (Z=52); the real one is Sn (Z=50), because magic-number shell effects are missing.

## Traceability
Every printed σ carries: dataset+version, isotope/MAT, MF/MT, stored energy range, units, interpolation law, source file + SHA-256, processing chain. Probabilities are tagged CALCULATION. Assumptions are listed separately. Data files are hash-checked on load.

## Rebuilding data (optional; processed data are included)
Needs NJOY2016 built from source (`gfortran`, `cmake`; commit/version recorded in `data/MANIFEST.json`):
`python scripts/build_data.py --njoy /path/to/njoy`. Other NJOY versions may change numbers slightly.

## Validation
Reference: V. F. Sears, *Neutron News* 3(3), 26–37 (1992), via https://ncnr.nist.gov/resources/n-lengths/list.html (page notes values were hand-entered; errors possible). Quantity: absorption at 2200 m/s (E = 0.0252989 eV), sum of evaluated fission + capture + charged-particle channels.

| Case | Result |
|---|---|
| B-10, Al-27, Ag-109 | AGREE within 3σ of the reference uncertainty (evaluation uncertainty is not propagated) |
| U-235 | **DISCREPANT**: +0.78 %, 4.8σ. Reported, not hidden. Causes not verified (newer evaluation vs 1992 compilation? 0 K definition?). A newer reference (Carlson et al., Nucl. Data Sheets 148, 143 (2018), Sec. 3.3) was not retrieved. |

Other checks: summation rule closes to ~1e-7; B-10 (n,α) follows 1/v within 0.2 % (1e-4–1 eV); parser output identical to the independent `endf` package; tampered data rejected; no extrapolation outside stored ranges.

## Limitations
- Only 4 isotopes. Anything else: "unsupported by available data".
- Uncertainties not propagated; covariance presence per channel is reported.
- Raw ENDF copies come from NJOY2016's test resources, not verified byte-for-byte against NNDC.
- 0 K, no Doppler broadening, no URR self-shielding, no thermal-scattering law; products and angular/energy distributions unsupported.
- Out of scope by design: chain reactions, criticality, multiplication, yield, weapons, handling/production of materials.

## Citations
- ENDF/B-VIII.0: D. A. Brown et al., Nucl. Data Sheets 148, 1 (2018), doi:10.1016/j.nds.2018.02.001
- NJOY2016: https://github.com/njoy/NJOY2016 (build 2016.80, commit in manifest)
- Sears (1992), as above. Neutron mass: CODATA 2018.

## Simulating isotopes without evaluated data (model estimate)

    neutronsim simulate Na-23 --energy 0.0253 -n 20000 --seed 1     # random draws from physics-law probabilities
    neutronsim forecast Na-23 --energy 0.0253                        # cross sections only

Works below 1 keV (elastic + capture) and from 1 keV to 20 MeV with the hf.py channels (elastic, inelastic, (n,2n), (n,p), (n,alpha), capture, ...,
each with its ENDF MT number so the visualizer can draw the emitted particles). Above 20 MeV only elastic + non-elastic. Refused where capture cannot
be predicted (compound nucleus too light or weakly bound) below 5 MeV. Always labelled as a
MODEL ESTIMATE; the printed probability ranges are the model's measured scatter, not guarantees.
The wizard offers the same draws (and the visualizer) for isotopes without data.

## One-command launchers (no installation needed)

    python run_gui.py                 # open the GUI wizard (isotope page has the periodic-table explorer; Results page has the visualizer)
    python run_gui.py explore U-235   # periodic-table explorer on its own
    python run_gui.py boxgui          # box test window (any isotope, density slider, live stats, animation tab)
    python run_gui.py boxviz Fe-56    # same window opened on the animation tab
    python run_all.py                 # run every check, example, simulation and the test suite
    python run_all.py --gui           # same, then open the GUI if everything passed
The GUI needs tkinter (Linux: `sudo apt install python3-tk`; the python.org installers for Windows/macOS include it).
