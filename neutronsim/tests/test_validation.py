import json
import pytest
from neutronsim import validation


def test_e2200():
    assert validation.e_2200_eV() == pytest.approx(0.0252989, abs=2e-7)


def test_reference_file_is_cited():
    spec = json.loads(validation.CASES_PATH.read_text())
    assert "Sears" in spec["reference"]["citation"] and spec["reference"]["retrieved_from"].startswith("https://")
    assert spec["reference"]["caveat"] and spec["k"] == 3
    assert all("u_b" in c and c["u_b"] > 0 for c in spec["cases"])


def test_validation_reports_expected_statuses_and_never_fails():
    results = validation.run()
    assert len(results) == 4
    assert {r.status for r in results} <= {"AGREES", "DISCREPANT"}   # no FAIL (gross-error bound)
    for r in results:
        assert r.status == r.expected_status, (r.id, r.status, r.sigma_units)


def test_known_discrepancy_is_reported_not_hidden():
    u235 = next(r for r in validation.run() if r.isotope == "U-235")
    assert u235.status == "DISCREPANT" and u235.sigma_units > 3 and u235.note


def test_classifier():
    # (diff, u_ref, ref, k, gross_rel): gross bound here is 0.02 * 100 = 2
    assert validation.classify(0.8, 0.3, 100, 3, 0.02) == "AGREES"       # 0.8 <= 3*0.3
    assert validation.classify(1.5, 0.3, 100, 3, 0.02) == "DISCREPANT"   # > 0.9 but <= 2
    assert validation.classify(-5.0, 0.3, 100, 3, 0.02) == "FAIL"        # beyond gross bound
    assert validation.classify(-0.75, 0.3, 100, 3, 0.02) == "AGREES"     # negative diffs handled
