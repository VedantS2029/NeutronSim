import json
import math
import shutil
import pytest
from conftest import ISOTOPES, get
from neutronsim import core

ENERGIES = [1e-5, 0.0253, 1.0, 1e3, 1e5, 1e6, 5e6, 1.4e7, 2e7]


@pytest.mark.parametrize("text,expected", [("U-235", "U-235"), ("u235", "U-235"), ("235U", "U-235"),
                                           ("b10", "B-10"), (" 27Al ", "Al-27"), ("ag-109", "Ag-109")])
def test_normalize_isotope(text, expected):
    assert core.normalize_isotope(text) == expected


@pytest.mark.parametrize("bad", ["", "uranium", "U--235", "235", "U235x", "1234U"])
def test_bad_isotope(bad):
    with pytest.raises(core.InputError):
        core.normalize_isotope(bad)


@pytest.mark.parametrize("val,unit", [(0, "eV"), (-1, "eV"), (float("nan"), "eV"), (float("inf"), "MeV"),
                                      ("abc", "eV"), (1, "GeV")])
def test_bad_energy(val, unit):
    with pytest.raises(core.InputError):
        core.to_eV(val, unit)


def test_energy_units():
    assert core.to_eV(1, "MeV") == 1e6 and core.to_eV(2, "keV") == 2e3


def test_unsupported_isotope_is_explicit():
    with pytest.raises(core.UnsupportedError, match="unsupported by available data"):
        core.Dataset("Pu-239")


@pytest.mark.parametrize("iso", ISOTOPES)
def test_no_extrapolation_outside_range(iso):
    ds = get(iso)
    lo, hi = ds.energy_range_eV
    for e in (lo / 10, hi * 1.0001, hi * 10):
        r = core.evaluate(ds, e)
        assert not r.supported and "unsupported by available data" in r.reason
        assert r.channels == [] and r.total is None      # no numbers invented
    with pytest.raises(core.UnsupportedError):
        core.sample_outcomes(core.evaluate(ds, hi * 10), 10)


@pytest.mark.parametrize("iso", ISOTOPES)
def test_summation_rule_closes_and_probabilities_valid(iso):
    ds = get(iso)
    lo, hi = ds.energy_range_eV
    for e in [x for x in ENERGIES if lo <= x <= hi]:
        r = core.evaluate(ds, e)
        assert abs(r.remainder_fraction) < 1e-3, (iso, e, r.remainder_fraction)
        ps = [c.probability for c in r.channels if c.probability is not None]
        assert all(0 <= p <= 1 for p in ps)
        assert sum(ps) == pytest.approx(1.0, abs=1e-3)


def test_threshold_channel_handling():
    ds = get("Al-27")
    low = core.evaluate(ds, 0.0253)
    c = next(c for c in low.channels if c.mt == 107)
    assert c.status == "below_threshold" and c.sigma_barn == 0.0 and "threshold" in c.note
    high = core.evaluate(ds, 5e6)
    c = next(c for c in high.channels if c.mt == 107)
    assert c.status == "ok" and c.sigma_barn > 0


def test_non_threshold_channel_below_range_is_unsupported_not_zero():
    # E below the stored range of a channel whose table starts at a nonzero value -> must not become 0
    ds = get("B-10")
    lo, _ = ds.energy_range_eV
    c = ds.channel(102, lo / 10)
    assert c.status == "unsupported" and c.sigma_barn is None


@pytest.mark.parametrize("iso", ISOTOPES)
def test_every_value_has_full_provenance(iso):
    ds = get(iso)
    r = core.evaluate(ds, 1e5 if iso != "B-10" else 1e3)
    required = {"dataset", "isotope", "mat", "file", "channel", "energy_range_eV", "units",
                "interpolation", "source_file", "source_sha256", "processed_file", "processing"}
    for c in [r.total] + r.channels:
        assert c.source is not None and required <= set(c.source)
        assert c.source["dataset"] == "ENDF/B-VIII.0" and c.source["units"] == "barn"
        assert c.source["processing"] and c.source["interpolation"] in {"lin-lin", "histogram", "lin-log",
                                                                       "log-lin", "log-log", None}


@pytest.mark.parametrize("iso", ISOTOPES)
def test_uncertainty_never_invented(iso):
    ds = get(iso)
    r = core.evaluate(ds, 1e3)
    for c in [r.total] + r.channels:
        assert c.uncertainty.startswith(("not propagated", "unavailable"))


def test_manifest_records_processing_and_caveat():
    man = json.loads((core.DATA_DIR / "MANIFEST.json").read_text())
    assert man["processing"]["code"].startswith("NJOY")
    assert "NOT compared" in man["source"]["caveat"]
    assert man["processing"]["not_applied"]
    for info in man["isotopes"].values():
        assert info["library"] == "ENDF/B-VIII.0" and len(info["raw_sha256"]) == 64


def test_tampered_data_detected(tmp_path):
    shutil.copytree(core.DATA_DIR / "pendf", tmp_path / "pendf")
    shutil.copy(core.DATA_DIR / "MANIFEST.json", tmp_path / "MANIFEST.json")
    f = tmp_path / "pendf" / "B-10.pendf.gz"
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(core.DataIntegrityError):
        core.Dataset("B-10", tmp_path)


def test_b10_alpha_follows_one_over_v_at_low_energy():
    """Behavioural check (textbook 1/v law), NOT a published numerical reference."""
    t = get("B-10").tabs[107]
    base = t(0.0253) * math.sqrt(0.0253)
    for e in (1e-4, 1e-3, 1e-2, 0.1, 1.0):
        assert t(e) * math.sqrt(e) == pytest.approx(base, rel=2e-3)


def test_fission_only_where_data_exist():
    assert 18 in get("U-235").tabs
    for iso in ("B-10", "Al-27", "Ag-109"):
        assert 18 not in get(iso).tabs
        assert all(c.mt != 18 for c in core.evaluate(get(iso), 1e6).channels)


# ---- sampling ----
def test_sampling_counts_and_reproducibility():
    r = core.evaluate(get("Ag-109"), 1e6)
    a = core.sample_outcomes(r, 5000, seed=7)
    b = core.sample_outcomes(r, 5000, seed=7)
    assert a == b and sum(x["count"] for x in a["rows"]) == 5000
    assert "NOT the uncertainty of the nuclear data" in a["error_meaning"]
    assert core.sample_outcomes(r, 5000, seed=8) != a


def test_sampling_converges_to_model_probabilities():
    r = core.evaluate(get("B-10"), 0.0253)
    s = core.sample_outcomes(r, 200_000, seed=1)
    for row in s["rows"]:
        assert abs(row["observed_fraction"] - row["model_probability"]) <= 5 * row["sampling_std_error"] + 1e-9


@pytest.mark.parametrize("n", [0, -5, core.MAX_SAMPLES + 1, 2.5])
def test_sampling_input_validation(n):
    r = core.evaluate(get("B-10"), 1.0)
    with pytest.raises(core.InputError):
        core.sample_outcomes(r, n)
