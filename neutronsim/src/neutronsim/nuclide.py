"""Step 1 of the isotope-explorer: the particle-count model (no GUI, stdlib only).

A Nuclide is just three whole numbers: protons (Z), neutrons (N) and electrons (e).

    protons   -> which ELEMENT it is          (change Z and you change the element)
    neutrons  -> which ISOTOPE of that element (change N and you change the isotope)
    electrons -> the ION CHARGE, charge = Z - e (change e and you make an ion, NOT a new isotope)

Nothing here predicts physics; it is bookkeeping and naming, so later steps (binding energy,
decay modes, real reaction data) have one well-tested object to work on.
Element symbols/names are standard IUPAC (Z = 1..118). Group/period layout follows the common
18-column table with Lu and Lr in group 3 (IUPAC's 2021 provisional report) and the f-block shown below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

# (symbol, name) indexed by Z-1
_ELEMENTS = (
    "H Hydrogen", "He Helium", "Li Lithium", "Be Beryllium", "B Boron", "C Carbon", "N Nitrogen", "O Oxygen",
    "F Fluorine", "Ne Neon", "Na Sodium", "Mg Magnesium", "Al Aluminium", "Si Silicon", "P Phosphorus",
    "S Sulfur", "Cl Chlorine", "Ar Argon", "K Potassium", "Ca Calcium", "Sc Scandium", "Ti Titanium",
    "V Vanadium", "Cr Chromium", "Mn Manganese", "Fe Iron", "Co Cobalt", "Ni Nickel", "Cu Copper", "Zn Zinc",
    "Ga Gallium", "Ge Germanium", "As Arsenic", "Se Selenium", "Br Bromine", "Kr Krypton", "Rb Rubidium",
    "Sr Strontium", "Y Yttrium", "Zr Zirconium", "Nb Niobium", "Mo Molybdenum", "Tc Technetium",
    "Ru Ruthenium", "Rh Rhodium", "Pd Palladium", "Ag Silver", "Cd Cadmium", "In Indium", "Sn Tin",
    "Sb Antimony", "Te Tellurium", "I Iodine", "Xe Xenon", "Cs Caesium", "Ba Barium", "La Lanthanum",
    "Ce Cerium", "Pr Praseodymium", "Nd Neodymium", "Pm Promethium", "Sm Samarium", "Eu Europium",
    "Gd Gadolinium", "Tb Terbium", "Dy Dysprosium", "Ho Holmium", "Er Erbium", "Tm Thulium", "Yb Ytterbium",
    "Lu Lutetium", "Hf Hafnium", "Ta Tantalum", "W Tungsten", "Re Rhenium", "Os Osmium", "Ir Iridium",
    "Pt Platinum", "Au Gold", "Hg Mercury", "Tl Thallium", "Pb Lead", "Bi Bismuth", "Po Polonium",
    "At Astatine", "Rn Radon", "Fr Francium", "Ra Radium", "Ac Actinium", "Th Thorium", "Pa Protactinium",
    "U Uranium", "Np Neptunium", "Pu Plutonium", "Am Americium", "Cm Curium", "Bk Berkelium",
    "Cf Californium", "Es Einsteinium", "Fm Fermium", "Md Mendelevium", "No Nobelium", "Lr Lawrencium",
    "Rf Rutherfordium", "Db Dubnium", "Sg Seaborgium", "Bh Bohrium", "Hs Hassium", "Mt Meitnerium",
    "Ds Darmstadtium", "Rg Roentgenium", "Cn Copernicium", "Nh Nihonium", "Fl Flerovium", "Mc Moscovium",
    "Lv Livermorium", "Ts Tennessine", "Og Oganesson",
)
ELEMENTS: tuple[tuple[str, str], ...] = tuple(tuple(s.split(" ", 1)) for s in _ELEMENTS)  # type: ignore[misc]
MAX_Z = len(ELEMENTS)                      # 118
SYMBOL_TO_Z = {sym: i + 1 for i, (sym, _) in enumerate(ELEMENTS)}
NAME_TO_Z = {name.lower(): i + 1 for i, (_, name) in enumerate(ELEMENTS)}
NAME_TO_Z["aluminum"], NAME_TO_Z["cesium"], NAME_TO_Z["sulphur"] = 13, 55, 16   # common spellings

MAX_A = 350            # generous upper bound on mass number; keeps typos like "U-2350" out
MAX_EXTRA_ELECTRONS = 3  # up to 3 more electrons than protons (real anions beyond -2 are essentially unknown)

_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_SPECIAL = {(1, 0): "protium", (1, 1): "deuterium", (1, 2): "tritium"}   # historical names for H-1/H-2/H-3


class NuclideError(ValueError):
    """Impossible or unparseable particle counts."""


def element_symbol(z: int) -> str:
    _check_z(z)
    return ELEMENTS[z - 1][0]


def element_name(z: int) -> str:
    _check_z(z)
    return ELEMENTS[z - 1][1]


def _check_int(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise NuclideError(f"{what} must be a whole number, got {v!r}")
    return v


def _check_z(z) -> None:
    _check_int(z, "proton count")
    if not 1 <= z <= MAX_Z:
        raise NuclideError(f"proton count must be 1..{MAX_Z} (H..Og), got {z}")


def periodic_position(z: int) -> tuple[int, int]:
    """(row, column) in the usual 18-column table. Rows 1-7 are the main table; the f-block
    (La-Yb, Ac-No) is drawn in rows 9 and 10, columns 3-16, below the main table (row 8 is a gap)."""
    _check_z(z)
    if z == 1:
        return 1, 1
    if z == 2:
        return 1, 18
    if z <= 4:
        return 2, z - 2
    if z <= 10:
        return 2, z + 8
    if z <= 12:
        return 3, z - 10
    if z <= 18:
        return 3, z
    if z <= 36:
        return 4, z - 18
    if z <= 54:
        return 5, z - 36
    if z <= 56:
        return 6, z - 54
    if z <= 70:
        return 9, z - 54            # La(57)->col 3 ... Yb(70)->col 16
    if z <= 86:
        return 6, z - 68            # Lu(71)->col 3 ... Rn(86)->col 18
    if z <= 88:
        return 7, z - 86
    if z <= 102:
        return 10, z - 86           # Ac(89)->col 3 ... No(102)->col 16
    return 7, z - 100               # Lr(103)->col 3 ... Og(118)->col 18


@dataclass(frozen=True)
class Nuclide:
    """An atom or ion described by its particle counts. Immutable: use with_* to get a modified copy."""
    protons: int
    neutrons: int
    electrons: int | None = None      # None means "neutral" (electrons = protons)

    def __post_init__(self):
        _check_z(self.protons)
        _check_int(self.neutrons, "neutron count")
        if self.neutrons < 0:
            raise NuclideError(f"neutron count cannot be negative, got {self.neutrons}")
        if self.protons + self.neutrons > MAX_A:
            raise NuclideError(f"mass number {self.protons + self.neutrons} exceeds the supported maximum {MAX_A}")
        e = self.protons if self.electrons is None else self.electrons
        _check_int(e, "electron count")
        if e < 0:
            raise NuclideError(f"electron count cannot be negative, got {e}")
        if e > self.protons + MAX_EXTRA_ELECTRONS:
            raise NuclideError(f"{e} electrons on {self.protons} protons (charge {self.protons - e}) is beyond "
                               f"the supported range (at most {MAX_EXTRA_ELECTRONS} extra electrons)")
        object.__setattr__(self, "electrons", e)   # normalise None -> Z

    # ---- identity -------------------------------------------------------------------------
    @property
    def Z(self) -> int:
        return self.protons

    @property
    def N(self) -> int:
        return self.neutrons

    @property
    def A(self) -> int:
        return self.protons + self.neutrons

    @property
    def charge(self) -> int:
        return self.protons - self.electrons

    @property
    def symbol(self) -> str:
        return element_symbol(self.protons)

    @property
    def element(self) -> str:
        return element_name(self.protons)

    @property
    def key(self) -> str:
        """Isotope key in the form the data layer uses, e.g. 'U-235'. Ignores electrons (the nucleus is the same)."""
        return f"{self.symbol}-{self.A}"

    @property
    def special_name(self) -> str | None:
        return _SPECIAL.get((self.protons, self.neutrons))

    @property
    def label(self) -> str:
        """Nuclear-notation label with charge, e.g. '²³⁵U', '¹⁶O²⁻', '¹H⁺'."""
        c = self.charge
        q = "" if c == 0 else (("+" if c > 0 else "-") if abs(c) == 1 else f"{abs(c)}{'+' if c > 0 else '-'}")
        return f"{str(self.A).translate(_SUP)}{self.symbol}{q.translate(_SUP)}"

    @property
    def kind(self) -> str:
        c = self.charge
        return "neutral atom" if c == 0 else ("cation (positive ion)" if c > 0 else "anion (negative ion)")

    @property
    def isotope_name(self) -> str:
        sn = f" ({self.special_name})" if self.special_name else ""
        return f"{self.element}-{self.A}{sn}"

    # ---- what changed, in plain words ----------------------------------------------------
    def describe(self) -> str:
        lines = [f"{self.label}  =  {self.isotope_name}, {self.kind}",
                 f"  protons {self.Z}  neutrons {self.N}  electrons {self.electrons}  "
                 f"(mass number A = {self.A}, charge {'0' if self.charge == 0 else format(self.charge, '+d')})"]
        if self.charge != 0:
            lines.append("  Charge comes only from the electron count; the nucleus (and so the isotope) is unchanged.")
        return "\n".join(lines)

    # ---- editing: each returns a NEW Nuclide ---------------------------------------------
    def with_neutrons(self, n: int) -> "Nuclide":
        """Different isotope of the same element (same Z, same electron count)."""
        return replace(self, neutrons=n)

    def with_protons(self, z: int, keep_neutral: bool = True) -> "Nuclide":
        """Different element. By default the result is neutral (electrons follow Z); keep_neutral=False keeps the electron count."""
        return replace(self, protons=z, electrons=None if keep_neutral else self.electrons)

    def with_electrons(self, e: int) -> "Nuclide":
        """Same nucleus, different ion charge. This does NOT change the isotope."""
        return replace(self, electrons=e)

    def neutralised(self) -> "Nuclide":
        return replace(self, electrons=None)

    # ---- parsing -------------------------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> "Nuclide":
        """Accepts 'U-235', 'u235', '235U', 'Uranium-235', 'deuterium', 'tritium'; optional charge suffix like '2+', '-', '3-'
        (e.g. 'O-16 2-', 'Fe56 3+'). The suffix sets the electron count; no suffix means neutral."""
        s = text.strip()
        if not s:
            raise NuclideError("empty isotope name")
        # A trailing "2+", "3-", "+" or "-" is a charge, but "H-1+" must not be read as "H-" + "1+": try the
        # longer suffix first and keep the first split whose remainder is a complete isotope.
        for pat in (r"(.*?)[\s^_]*(\d{1,3})([+-])", r"(.*?)[\s^_]*()([+-])"):
            m = re.fullmatch(pat, s)
            if m and m.group(1).strip():
                try:
                    base = cls._parse_neutral(m.group(1))
                except NuclideError:
                    continue
                charge = int(m.group(2) or 1) * (1 if m.group(3) == "+" else -1)
                return base.with_electrons(base.Z - charge)
        return cls._parse_neutral(s)

    @classmethod
    def _parse_neutral(cls, text: str) -> "Nuclide":
        s = text.strip()
        low = s.lower()
        for (z, n), nm in _SPECIAL.items():
            if low == nm:
                return cls(z, n)
        if low == "deuteron":
            return cls(1, 1)
        for pat, order in ((r"([A-Za-z]+)(?:\s+|-)?(\d{1,3})", "sym_a"), (r"(\d{1,3})(?:\s+|-)?([A-Za-z]+)", "a_sym")):
            m = re.fullmatch(pat, s)
            if m:
                sym, a = (m.group(1), m.group(2)) if order == "sym_a" else (m.group(2), m.group(1))
                z = SYMBOL_TO_Z.get(sym[0].upper() + sym[1:].lower()) if len(sym) <= 2 else NAME_TO_Z.get(sym.lower())
                if z is None:
                    raise NuclideError(f"unknown element {sym!r}")
                a = int(a)
                if a < z:
                    raise NuclideError(f"mass number {a} is smaller than the proton count {z} for {element_symbol(z)}")
                return cls(z, a - z)
        raise NuclideError(f"cannot read {text!r} as an isotope (try 'U-235', '235U' or 'Uranium-235')")
