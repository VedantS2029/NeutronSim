import pytest
from neutronsim import core
from neutronsim.nuclide import (ELEMENTS, MAX_Z, Nuclide, NuclideError, element_name, element_symbol,
                                periodic_position)


def test_table_is_complete_and_unique():
    assert MAX_Z == 118 == len(ELEMENTS)
    assert len({s for s, _ in ELEMENTS}) == 118 and len({n for _, n in ELEMENTS}) == 118
    assert [element_symbol(z) for z in (1, 2, 6, 8, 26, 79, 82, 92, 118)] == ["H", "He", "C", "O", "Fe", "Au", "Pb", "U", "Og"]
    assert element_name(92) == "Uranium" and element_name(74) == "Tungsten"


def test_periodic_positions_unique_and_known():
    pos = {z: periodic_position(z) for z in range(1, 119)}
    assert len(set(pos.values())) == 118                       # no two elements share a cell
    known = {1: (1, 1), 2: (1, 18), 3: (2, 1), 10: (2, 18), 11: (3, 1), 18: (3, 18), 19: (4, 1), 26: (4, 8),
             36: (4, 18), 47: (5, 11), 54: (5, 18), 55: (6, 1), 56: (6, 2), 57: (9, 3), 70: (9, 16), 71: (6, 3),
             79: (6, 11), 86: (6, 18), 87: (7, 1), 89: (10, 3), 92: (10, 6), 102: (10, 16), 103: (7, 3), 118: (7, 18)}
    for z, p in known.items():
        assert pos[z] == p, z


def test_counts_and_charge():
    u = Nuclide(92, 143)
    assert (u.A, u.charge, u.electrons, u.key, u.label) == (235, 0, 92, "U-235", "²³⁵U")
    o = Nuclide(8, 8, 10)
    assert (o.charge, o.label, o.kind) == (-2, "¹⁶O²⁻", "anion (negative ion)")
    assert Nuclide(1, 0, 0).label == "¹H⁺" and Nuclide(26, 30, 23).label == "⁵⁶Fe³⁺"


def test_electrons_change_charge_not_isotope():
    a = Nuclide(6, 6)
    b = a.with_electrons(3)
    assert b.key == a.key == "C-12" and b.charge == 3 and "unchanged" in b.describe()


def test_neutrons_make_isotopes_protons_make_elements():
    c12 = Nuclide(6, 6)
    assert c12.with_neutrons(8).key == "C-14"
    n14 = c12.with_protons(7)
    assert n14.symbol == "N" and n14.electrons == 7 and n14.neutrons == 6
    cation = Nuclide(6, 6, 4)
    assert cation.with_neutrons(8).charge == 2                 # electron count preserved
    assert cation.with_protons(7).charge == 0                  # default: neutral
    assert cation.with_protons(7, keep_neutral=False).charge == 3
    assert cation.neutralised().charge == 0


def test_special_names():
    assert Nuclide(1, 1).isotope_name == "Hydrogen-2 (deuterium)"
    assert Nuclide(1, 2).isotope_name == "Hydrogen-3 (tritium)" and Nuclide(8, 8).isotope_name == "Oxygen-16"


@pytest.mark.parametrize("text,expected", [
    ("U-235", (92, 143, 92)), ("u235", (92, 143, 92)), ("235U", (92, 143, 92)), ("Uranium-235", (92, 143, 92)),
    ("b10", (5, 5, 5)), ("27 Al", (13, 14, 13)), ("deuterium", (1, 1, 1)), ("Tritium", (1, 2, 1)),
    ("O-16 2-", (8, 8, 10)), ("Fe56 3+", (26, 30, 23)), ("H-1+", (1, 0, 0)), ("C-12 -", (6, 6, 7)),
    ("Aluminum-27", (13, 14, 13)),
    ("U-238 91+", (92, 146, 1)), ("Fe-56 26+", (26, 30, 0)), ("H-1 +", (1, 0, 0)),
])
def test_parse(text, expected):
    n = Nuclide.parse(text)
    assert (n.Z, n.N, n.electrons) == expected


@pytest.mark.parametrize("bad", ["", "  ", "Xx-12", "U", "235", "U-5", "H-0", "Uranium", "U--235", "U-2350", "1234U"])
def test_parse_rejects(bad):
    with pytest.raises(NuclideError):
        Nuclide.parse(bad)


@pytest.mark.parametrize("args", [(0, 1), (119, 1), (-1, 5), (6, -1), (6, 6, -1), (6, 6, 10), (6, 400), (6.0, 6), (True, 6), (6, "6")])
def test_impossible_counts_rejected(args):
    with pytest.raises(NuclideError):
        Nuclide(*args)


@pytest.mark.parametrize("iso", ["B-10", "Al-27", "Ag-109", "U-235"])
def test_key_matches_data_layer(iso):
    assert Nuclide.parse(iso).key == core.normalize_isotope(iso)
