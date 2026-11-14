import json
import pytest
from neutronsim import cli


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_xs_output_is_labelled_and_traceable(capsys):
    code, out, _ = run(capsys, "xs", "B-10", "--energy", "0.0253")
    assert code == 0
    for s in ("EDUCATIONAL MODEL", "DATA", "CALCULATION", "UNCERTAINTY", "SOURCES", "ASSUMPTIONS",
              "ENDF/B-VIII.0", "sha256", "MF3/MT107", "barn", "NJOY"):
        assert s in out
    assert "Radioactive decay: unsupported by available data" in out


def test_energy_is_required(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["xs", "B-10"])
    assert e.value.code == 2
    assert "--energy" in capsys.readouterr().err


def test_unsupported_isotope_and_energy(capsys):
    code, _, err = run(capsys, "xs", "Pu-239", "--energy", "1")
    assert code == 2 and "unsupported by available data" in err
    code, out, _ = run(capsys, "xs", "B-10", "--energy", "100", "--unit", "MeV")
    assert code == 3 and "unsupported by available data" in out and "No extrapolation" in out


@pytest.mark.parametrize("args", [("--energy", "-1"), ("--energy", "0"), ("--energy", "nan")])
def test_bad_energy_rejected(capsys, args):
    code, _, err = run(capsys, "xs", "B-10", *args)
    assert code == 2 and "error" in err


def test_json_has_provenance_and_disclaimer(capsys):
    code, out, _ = run(capsys, "xs", "Ag-109", "--energy", "1", "--unit", "keV", "--json")
    d = json.loads(out)
    assert code == 0 and "EDUCATIONAL MODEL" in d["disclaimer"]
    assert d["total"]["source"]["dataset"] == "ENDF/B-VIII.0"


def test_sample_output_states_it_is_not_a_prediction(capsys):
    code, out, _ = run(capsys, "sample", "U-235", "--energy", "1", "--unit", "MeV", "-n", "2000", "--seed", "3")
    assert code == 0 and "do not predict" in out and "NOT the uncertainty" in out


def test_validate_and_explain_and_list(capsys):
    code, out, _ = run(capsys, "validate")
    assert code == 0 and "DISCREPANT" in out and "AGREES" in out
    code, out, _ = run(capsys, "explain", "fusion")
    assert "NOT modelled" in out
    code, out, _ = run(capsys, "list")
    assert "unsupported by available data" in out and "U-235" in out


def test_scope_exclusions_not_implemented():
    # fission is reported as a probability channel only; no multiplication/criticality API exists
    import neutronsim.core as c
    names = " ".join(dir(c)).lower()
    for banned in ("nubar", "keff", "k_eff", "critical", "multiplication", "yield", "chain"):
        assert banned not in names
