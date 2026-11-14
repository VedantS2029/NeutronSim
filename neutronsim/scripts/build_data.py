#!/usr/bin/env python3
"""Build data/pendf/*.pendf.gz and data/MANIFEST.json.

Pipeline (every step recorded in the manifest):
  1. download evaluated ENDF-6 file from a pinned GitHub commit; verify SHA-256
  2. read the file's own MF1/MT451 header (library/version come from the file, not the filename)
  3. run NJOY2016 RECONR (resonance reconstruction + linearisation), T = 0 K, err = 0.001
  4. keep MF1/MT451 + the MF3 sections for channels this program uses (no numerical change)

Usage: python scripts/build_data.py --njoy /path/to/njoy [--raw-dir data/raw]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neutronsim import endf  # noqa: E402
from neutronsim.channels import is_summation_member  # noqa: E402

NJOY_REPO_COMMIT = "ab27c6414a0f25b173f2b637e6dc7e428fdaff00"
RAW_URL = ("https://raw.githubusercontent.com/njoy/NJOY2016/" + NJOY_REPO_COMMIT
           + "/tests/resources/{name}")
RECONR_ERR = 0.001
RECONR_TEMP_K = 0.0

# id, source file, pinned SHA-256 of that file
ISOTOPES = [
    ("B-10", "n-005_B_010-ENDF8.0.endf", "a842bdac422194dd7dfe6b150e085c7610ba4d503ac27b2813b783e7117e678b"),
    ("Al-27", "n-013_Al_027-ENDF8.0.endf", "e1b2b359ae4ac39dd95c4f1de22f8828c8b0e83597565b62e8a63b3b9fcf32ea"),
    ("Ag-109", "n-047_Ag_109-ENDF8.0.endf", "3f0685971666330e55487e3db56d809e5c0106d7d5d2e6f3516346266f2aff5a"),
    ("U-235", "n-092_U_235-ENDF8.0.endf", "199fb786ec888485a83b4de1f6fecfff2de0096479bde282e6e418153b9f3b79"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, expected: str, raw_dir: Path) -> Path:
    dest = raw_dir / name
    if not dest.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {name}")
        urllib.request.urlretrieve(RAW_URL.format(name=name), dest)
    got = sha256(dest)
    if got != expected:
        raise SystemExit(f"SHA-256 mismatch for {name}: expected {expected}, got {got}")
    return dest


def run_reconr(njoy: Path, endf_file: Path, mat: int) -> tuple[Path, str, Path]:
    work = Path(tempfile.mkdtemp(prefix="njoy_"))
    shutil.copy(endf_file, work / "tape20")
    deck = (f"reconr\n20 21\n'RECONR for MAT {mat}'/\n{mat} 0 0/\n"
            f"{RECONR_ERR} {RECONR_TEMP_K}/\n0/\nstop\n")
    res = subprocess.run([str(njoy)], input=deck, text=True, cwd=work,
                         capture_output=True, timeout=1800)
    out = work / "tape21"
    if res.returncode != 0 or not out.exists():
        raise SystemExit(f"NJOY failed for MAT {mat}:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    m = re.search(r"njoy\s+(\S+)\s+(\S+)", res.stdout)
    version = f"NJOY {m.group(1)} ({m.group(2)})" if m else "NJOY (version not parsed)"
    return out, version, work


def extract(pendf: Path, dest_gz: Path) -> list[int]:
    """Keep MF1/MT451 and MF3 sections for channels in use. Lines are copied verbatim;
    only the SEND/FEND/MEND/TEND terminator records are regenerated so the result is a valid tape."""
    kept: list[int] = []

    def term(mat: int, mf: int, mt: int, ns: int) -> str:
        return f"{'':66}{mat:4d}{mf:2d}{mt:3d}{ns:5d}\n"

    with open(pendf, encoding="ascii") as fh:
        tpid = fh.readline()
    mat_id = 0
    with gzip.open(dest_gz, "wt", encoding="ascii") as out:
        out.write(tpid if tpid.endswith("\n") else tpid + "\n")
        mf3_written = False
        for (mat, mf, mt), lines in endf.iter_sections(pendf):
            mat_id = mat
            if mf == 1 and mt == 451:
                out.write("\n".join(lines) + "\n" + term(mat, 1, 0, 99999) + term(mat, 0, 0, 0))
            elif mf == 3 and (mt in (1, 2, 3) or is_summation_member(mt)):
                out.write("\n".join(lines) + "\n" + term(mat, 3, 0, 99999))
                kept.append(mt)
                mf3_written = True
        if mf3_written:
            out.write(term(mat_id, 0, 0, 0))
        out.write(term(0, 0, 0, 0) + term(-1, 0, 0, 0))
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--njoy", required=True, type=Path)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()

    (ROOT / "data" / "pendf").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "processing": {
            "code": None,
            "reconr": {"err": RECONR_ERR, "temperature_K": RECONR_TEMP_K},
            "not_applied": [
                "Doppler broadening (BROADR)",
                "unresolved-resonance self-shielding / probability tables (UNRESR, PURR)",
                "thermal scattering law S(alpha,beta) / chemical binding",
            ],
            "interpolation_in_program": "as stored in each MF3 record (RECONR output is lin-lin)",
        },
        "source": {
            "description": "Copies of ENDF/B evaluated files distributed in NJOY2016 tests/resources",
            "repository": "https://github.com/njoy/NJOY2016",
            "commit": NJOY_REPO_COMMIT,
            "caveat": ("Bytes were NOT compared with the official NNDC distribution "
                       "(NNDC was unreachable when these data were built). Library/version "
                       "below is read from each file's own MF1/MT451 flags."),
        },
        "isotopes": {},
    }

    for iso, name, expected in ISOTOPES:
        print(iso)
        raw = fetch(name, expected, args.raw_dir)
        hdr = endf.read_header(raw)
        pendf, version, work = run_reconr(args.njoy, raw, hdr.mat)
        manifest["processing"]["code"] = version
        dest = ROOT / "data" / "pendf" / f"{iso}.pendf.gz"
        kept = extract(pendf, dest)
        manifest["isotopes"][iso] = {
            "za": int(hdr.za),
            "mat": hdr.mat,
            "awr": hdr.awr,
            "library": hdr.library,
            "header_flags": {"NLIB": hdr.nlib, "NVER": hdr.nver, "LREL": hdr.lrel, "NSUB": hdr.nsub},
            "evaluation_title": hdr.description[0].strip() if hdr.description else "",
            "evaluation_emax_eV": hdr.emax,
            "raw_file": name,
            "raw_url": RAW_URL.format(name=name),
            "raw_sha256": expected,
            "resonance_parameters_in_source (MF2)": hdr.has_section(2),
            "covariance_sections_in_source": {
                "MF31 (nubar)": hdr.mts_in(31), "MF32 (resonance)": hdr.mts_in(32),
                "MF33 (cross sections)": hdr.mts_in(33),
            },
            "pendf_file": f"pendf/{iso}.pendf.gz",
            "pendf_sha256": sha256(dest),
            "mf3_sections_kept": sorted(kept),
        }
        shutil.rmtree(work, ignore_errors=True)
        print(f"  {hdr.library}, kept MT {sorted(kept)}")

    (ROOT / "data" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote data/MANIFEST.json")


if __name__ == "__main__":
    main()
