"""Wizard logic tests. No display or tkinter needed: only WizardModel is exercised."""
import json
import pytest
from neutronsim import core, gui


def model(**kw):
    base = dict(acknowledged=True, isotope="B-10", energy_text="0.0253", unit="eV")
    base.update(kw)
    return gui.WizardModel(**base)


def test_steps_gate_on_input():
    m = gui.WizardModel()
    assert m.check_acknowledged() and m.check_isotope()
    m.acknowledged, m.isotope = True, "B-10"
    assert m.check_acknowledged() is None and m.check_isotope() is None


@pytest.mark.parametrize("text", ["", "abc", "-1", "0", "nan", "inf"])
def test_bad_energy_text_is_reported(text):
    assert model(energy_text=text).check_energy()


def test_energy_outside_range_refused_not_extrapolated():
    msg = model(energy_text="100", unit="MeV").check_energy()
    assert msg and "No extrapolation" in msg


def test_isotope_without_data_runs_as_labelled_model_estimate():
    m = model(isotope="C-14", energy_text="0.0253")
    assert m.check_isotope() is None and not m.has_data
    out = m.run()
    assert "MODEL ESTIMATE" in out.text and "NO EVALUATED DATA" in out.text and not out.supported
    assert "model_estimate" in out.json_text


def test_garbage_isotope_still_rejected():
    assert model(isotope="Xx-999").check_isotope()


def test_estimate_draws_where_predicted_refuses_where_not_and_absurd_energy():
    assert model(isotope="Na-23", mode="sample", n_text="100").check_mode() is None                   # thermal: elastic + capture
    assert "capture" in model(isotope="C-14", mode="sample").check_mode()                             # C-15 too weakly bound for the model
    assert model(isotope="C-14", mode="sample", energy_text="14", unit="MeV").check_mode() is None     # fast: elastic + nonelastic
    assert model(isotope="Na-23", mode="sample", energy_text="1", unit="MeV").check_mode() is None   # 1 keV - 20 MeV: Hauser-Feshbach channels
    assert model(isotope="C-14", mode="sample", energy_text="14", unit="MeV").check_mode() is None     # too light for hf.py: geometric split from 5 MeV
    assert "light" in model(isotope="H-1", mode="sample", energy_text="1", unit="MeV").check_mode()
    assert "capture" in model(isotope="H-1", mode="sample").check_mode()                              # model cannot predict it
    assert model(isotope="C-14", energy_text="1", unit="GeV").check_energy()


def test_data_isotope_unchanged_by_estimate_path():
    out = model(isotope="B-10").run()
    assert out.supported and "MODEL ESTIMATE" not in out.text


@pytest.mark.parametrize("n,seed", [("0", ""), ("1.5", ""), ("2000000", ""), ("10", "x")])
def test_sampling_validation(n, seed):
    assert model(mode="sample", n_text=n, seed_text=seed).check_mode()


def test_run_matches_cli_output_and_labels(capsys):
    from neutronsim import cli
    out = model().run()
    cli.main(["xs", "B-10", "--energy", "0.0253", "--unit", "eV"])
    assert out.text.strip() == capsys.readouterr().out.strip()
    assert out.supported and "EDUCATIONAL MODEL" in out.text


def test_sample_is_seed_reproducible_and_json_valid():
    a = model(mode="sample", n_text="500", seed_text="1").run()
    b = model(mode="sample", n_text="500", seed_text="1").run()
    assert a.text == b.text and "random draws" in a.text
    assert "sample" in json.loads(a.json_text)


def test_run_refuses_without_acknowledgement():
    with pytest.raises(core.InputError):
        model(acknowledged=False).run()


def test_gui_cli_command_registered():
    from neutronsim import cli
    assert cli.build_parser().parse_args(["gui"]).cmd == "gui"
