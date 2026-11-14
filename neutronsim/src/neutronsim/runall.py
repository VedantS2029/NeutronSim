"""One command that exercises everything: `neutronsim all` (or `python run_all.py` from the project folder).

Runs the validation and model checks, a real-data example, model-estimate examples (including the refusals, which are
expected behaviour), and the test suite when pytest is installed. Prints a pass/fail table; exit code 0 only if all pass.
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import time
from pathlib import Path

# (label, argv, expected exit code)
STEPS = [
    ("validate against published reference values", ["validate"], 0),
    ("model check (masses, chemistry)", ["modelcheck"], 0),
    ("forecast check vs real data", ["forecastcheck"], 0),
    ("real data: cross sections U-235 thermal", ["xs", "U-235", "--energy", "0.0253"], 0),
    ("real data: random draws B-10 thermal", ["sample", "B-10", "--energy", "0.0253", "-n", "5000", "--seed", "1"], 0),
    ("no data: forecast Na-23 thermal", ["forecast", "Na-23", "--energy", "0.0253"], 0),
    ("no data: simulate Na-23 thermal", ["simulate", "Na-23", "--energy", "0.0253", "-n", "5000", "--seed", "1"], 0),
    ("no data: simulate Pb-208 at 14 MeV", ["simulate", "Pb-208", "--energy", "14", "--unit", "MeV", "-n", "5000", "--seed", "1"], 0),
    ("no data: simulate Na-23 at 1 MeV", ["simulate", "Na-23", "--energy", "1", "--unit", "MeV", "-n", "5000", "--seed", "1"], 0),
    ("no data: simulate H-1 at 1 MeV (must be refused: too light)", ["simulate", "H-1", "--energy", "1", "--unit", "MeV"], 2),
    ("box test: B-10 thermal, 1% of solid density", ["box", "B-10", "--energy", "0.0253", "--density-factor", "0.01", "-n", "1000", "--seed", "1"], 0),
    ("box test: Al-27 at 14 MeV", ["box", "Al-27", "--energy", "14", "--unit", "MeV", "-n", "500", "--seed", "1"], 0),
    ("box test: U-235 (must be refused: fission is out of scope)", ["box", "U-235", "--energy", "1", "--unit", "MeV"], 2),
    ("no data: predict C-14", ["predict", "C-14"], 0),
]


def _run_cli(argv: list[str]) -> tuple[int, str]:
    from . import cli
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as e:
            code = int(e.code or 0)
        except Exception as exc:  # a crash is a failure, shown, not hidden
            code = 99
            err.write(f"crash: {exc!r}\n")
    return code, out.getvalue() + err.getvalue()


def _run_tests() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[2]
    if not (root / "tests").is_dir():
        return "SKIP", "no tests folder next to the package"
    try:
        import pytest  # noqa: F401
    except ImportError:
        return "SKIP", "pytest is not installed (pip install pytest)"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(root / "tests")],
                       cwd=root, capture_output=True, text=True)
    return ("PASS" if p.returncode == 0 else "FAIL"), p.stdout + p.stderr


def main(verbose: bool = False, run_tests: bool = True, gui: bool = False) -> int:
    rows, failed = [], 0
    for label, argv, want in STEPS:
        t = time.time()
        code, text = _run_cli(argv)
        ok = code == want
        failed += not ok
        rows.append((label, "PASS" if ok else f"FAIL (exit {code}, expected {want})", time.time() - t))
        if verbose or not ok:
            print(f"\n----- neutronsim {' '.join(argv)} -----\n{text}")
    if run_tests:
        t = time.time()
        status, text = _run_tests()
        failed += status == "FAIL"
        rows.append(("test suite (pytest)", status + ("" if status == "PASS" else ""), time.time() - t))
        if verbose or status != "PASS":
            print("\n----- test suite -----\n" + text)
    print("\n" + "=" * 78 + "\nSUMMARY")
    for label, status, secs in rows:
        print(f"  {status:<30} {label}  ({secs:.1f}s)")
    print("=" * 78)
    print("ALL PASSED" if not failed else f"{failed} STEP(S) FAILED")
    if gui:
        if failed:
            print("Not opening the GUI because a step failed (see above). Open it anyway with:  python run_gui.py")
            return 1
        print("Opening the GUI (wizard; use 'Explore any isotope' on the isotope page for the periodic table, 'Visualize' / 'Box test' on the Results page, or Tools > Box test) ...")
        from . import gui as gui_mod
        return gui_mod.main()
    return 1 if failed else 0
