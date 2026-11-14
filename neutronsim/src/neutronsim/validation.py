"""Validation against published reference cases (see validation/reference_cases.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import channels, core

CASES_PATH = Path(__file__).resolve().parents[2] / "validation" / "reference_cases.json"
M_N_KG = 1.67492749804e-27   # CODATA 2018 neutron mass
EV_J = 1.602176634e-19       # exact


def e_2200_eV() -> float:
    return 0.5 * M_N_KG * 2200.0 ** 2 / EV_J


@dataclass
class CaseResult:
    id: str
    isotope: str
    evaluated_b: float
    reference_b: float
    reference_u_b: float
    diff_b: float
    diff_rel: float
    sigma_units: float
    status: str
    expected_status: str
    note: str
    parts: dict


def absorption(ds: core.Dataset, e_eV: float) -> tuple[float, dict]:
    parts = {}
    for mt in ds.partial_mts:
        if mt == 18 or 102 <= mt <= 117:
            c = ds.channel(mt, e_eV)
            if c.sigma_barn is not None:
                parts[mt] = c.sigma_barn
    return sum(parts.values()), parts


def classify(diff: float, u: float, ref: float, k: float, gross_rel: float) -> str:
    if abs(diff) <= k * u:
        return "AGREES"
    if abs(diff) <= gross_rel * ref:
        return "DISCREPANT"
    return "FAIL"


def run(data_dir: Path = core.DATA_DIR) -> list[CaseResult]:
    spec = json.loads(CASES_PATH.read_text())
    e = e_2200_eV()
    out = []
    for case in spec["cases"]:
        ds = core.Dataset(case["isotope"], data_dir)
        val, parts = absorption(ds, e)
        diff = val - case["value_b"]
        out.append(CaseResult(
            case["id"], case["isotope"], val, case["value_b"], case["u_b"], diff,
            diff / case["value_b"], diff / case["u_b"],
            classify(diff, case["u_b"], case["value_b"], spec["k"], spec["gross_error_bound_rel"]),
            case["expected_status"], case.get("known_discrepancy_note", ""), parts))
    return out
