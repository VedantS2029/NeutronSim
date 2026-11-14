"""Minimal ENDF-6 reader: MF1/MT451 header and MF3 (TAB1) cross-section sections.

Stdlib only. Reads data exactly as stored; no unit conversion, no smoothing.
ENDF-6 convention: energies in eV, cross sections in barns.
"""
from __future__ import annotations

import bisect
import gzip
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

_FLOAT_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$")


class DataError(Exception):
    """Malformed or unsupported evaluated-data content."""


def open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="ascii", errors="replace")
    return open(path, "rt", encoding="ascii", errors="replace")


def parse_float(field_text: str) -> float:
    """ENDF float: '1.302700+4' means 1.3027e4; plain floats also accepted."""
    s = field_text.strip()
    if not s:
        return 0.0
    m = _FLOAT_RE.match(s)
    if m:
        return float(m.group(1) + "e" + m.group(2))
    try:
        return float(s)
    except ValueError as exc:
        raise DataError(f"bad ENDF float {field_text!r}") from exc


def parse_int(field_text: str) -> int:
    s = field_text.strip()
    return int(s) if s else 0


def _ids(line: str):
    """(MAT, MF, MT) of an 80-column ENDF line, or None for non-data lines."""
    if len(line) < 75:
        return None
    try:
        return int(line[66:70]), int(line[70:72]), int(line[72:75])
    except ValueError:
        return None


def iter_sections(path):
    """Yield ((mat, mf, mt), [lines]) for every section in a file."""
    cur_key, buf = None, []
    with open_text(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            key = _ids(line)
            if key is None or key[1] == 0 or key[2] == 0:  # SEND/FEND/MEND/TEND
                if cur_key is not None and buf:
                    yield cur_key, buf
                cur_key, buf = None, []
                continue
            if key != cur_key:
                if cur_key is not None and buf:
                    yield cur_key, buf
                cur_key, buf = key, []
            buf.append(line)
    if cur_key is not None and buf:
        yield cur_key, buf


def _fields(line: str):
    return [line[i : i + 11] for i in range(0, 66, 11)]


def _cont(line: str):
    f = _fields(line)
    return (
        parse_float(f[0]), parse_float(f[1]),
        parse_int(f[2]), parse_int(f[3]), parse_int(f[4]), parse_int(f[5]),
    )


@dataclass
class Header:
    mat: int
    za: float
    awr: float
    nlib: int
    nmod: int
    lrp: int
    lfi: int
    lrel: int
    nsub: int
    nver: int
    emax: float
    temp: float
    description: list[str]
    directory: list[tuple[int, int, int]]  # (MF, MT, n_lines)

    @property
    def library(self) -> str:
        """Library name derived from header flags. Unknown combinations stay explicit."""
        if self.nlib == 0 and self.nver == 8:
            return f"ENDF/B-VIII.{self.lrel}"
        if self.nlib == 0 and self.nver == 7:
            return f"ENDF/B-VII.{self.lrel}"
        return f"unrecognised (NLIB={self.nlib}, NVER={self.nver}, LREL={self.lrel})"

    def has_section(self, mf: int, mt: int | None = None) -> bool:
        return any(m == mf and (mt is None or t == mt) for m, t, _ in self.directory)

    def mts_in(self, mf: int) -> list[int]:
        return sorted(t for m, t, _ in self.directory if m == mf)


def read_header(path) -> Header:
    for (mat, mf, mt), lines in iter_sections(path):
        if mf == 1 and mt == 451:
            za, awr, lrp, lfi, nlib, nmod = _cont(lines[0])
            _, _, _, _, _, _nfor = _cont(lines[1])
            _awi, emax, lrel, _, nsub, nver = _cont(lines[2])
            temp, _, _ldrv, _, nwd, nxc = _cont(lines[3])
            desc = [ln[:66].rstrip() for ln in lines[4 : 4 + nwd]]
            directory = []
            for ln in lines[4 + nwd : 4 + nwd + nxc]:
                f = _fields(ln)
                directory.append((parse_int(f[2]), parse_int(f[3]), parse_int(f[4])))
            return Header(mat, za, awr, nlib, nmod, lrp, lfi, lrel, nsub, nver,
                          emax, temp, desc, directory)
    raise DataError(f"no MF1/MT451 header found in {path}")


# ENDF interpolation laws
HISTOGRAM, LINLIN, LINLOG, LOGLIN, LOGLOG = 1, 2, 3, 4, 5


@dataclass
class Tab1:
    """Tabulated function y(x): x = energy (eV), y = cross section (b)."""
    mt: int
    qm: float
    qi: float
    lr: int
    nbt: list[int]
    laws: list[int]
    x: list[float]
    y: list[float]
    _xs: list[float] = field(init=False, repr=False)

    def __post_init__(self):
        self._xs = self.x

    @property
    def e_min(self) -> float:
        return self.x[0]

    @property
    def e_max(self) -> float:
        return self.x[-1]

    def law_at(self, i: int) -> int:
        """Law for the interval [x[i], x[i+1]] (0-based i)."""
        for nbt, law in zip(self.nbt, self.laws):
            if i + 1 < nbt:  # NBT is the 1-based index of the last point of a range
                return law
        return self.laws[-1]

    def __call__(self, e: float) -> float:
        """Interpolate at e (eV). Never extrapolates; raises outside the stored range."""
        if not (self.e_min <= e <= self.e_max):
            raise DataError(
                f"E={e:g} eV outside stored range [{self.e_min:g}, {self.e_max:g}] eV for MT={self.mt}"
            )
        i = bisect.bisect_right(self._xs, e) - 1  # right-continuous at repeated energies
        if i >= len(self.x) - 1:
            return self.y[-1]
        x0, x1, y0, y1 = self.x[i], self.x[i + 1], self.y[i], self.y[i + 1]
        if x1 == x0:
            return y1
        law = self.law_at(i)
        if law == HISTOGRAM:
            return y0
        if law == LINLIN:
            return y0 + (y1 - y0) * (e - x0) / (x1 - x0)
        if law == LINLOG:
            return y0 + (y1 - y0) * math.log(e / x0) / math.log(x1 / x0)
        if law in (LOGLIN, LOGLOG):
            if y0 <= 0 or y1 <= 0:
                raise DataError(f"log interpolation of non-positive values (MT={self.mt}, E={e:g})")
            if law == LOGLIN:
                return math.exp(math.log(y0) + (math.log(y1) - math.log(y0)) * (e - x0) / (x1 - x0))
            return math.exp(math.log(y0) + (math.log(y1) - math.log(y0)) * math.log(e / x0) / math.log(x1 / x0))
        raise DataError(f"unsupported interpolation law {law} (MT={self.mt})")


def _numbers(lines: list[str]) -> list[float]:
    out: list[float] = []
    for ln in lines:
        out.extend(parse_float(f) for f in _fields(ln))
    return out


def parse_mf3(lines: list[str], mt: int) -> Tab1:
    """Parse one MF3 section (HEAD + TAB1)."""
    qm, qi, _l1, lr, nr, np_ = _cont(lines[1])
    n_interp_lines = (nr + 2) // 3
    n_data_lines = (np_ + 2) // 3
    need = 2 + n_interp_lines + n_data_lines
    if len(lines) < need:
        raise DataError(f"MF3/MT{mt}: truncated section")
    interp = _numbers(lines[2 : 2 + n_interp_lines])[: 2 * nr]
    nbt = [int(v) for v in interp[0::2]]
    laws = [int(v) for v in interp[1::2]]
    data = _numbers(lines[2 + n_interp_lines : need])[: 2 * np_]
    x, y = data[0::2], data[1::2]
    if any(b < a for a, b in zip(x, x[1:])):
        raise DataError(f"MF3/MT{mt}: energy grid not monotonic")
    return Tab1(mt, qm, qi, lr, nbt, laws, x, y)


def read_mf3(path, mts: set[int] | None = None) -> dict[int, Tab1]:
    out: dict[int, Tab1] = {}
    for (mat, mf, mt), lines in iter_sections(path):
        if mf == 3 and (mts is None or mt in mts):
            out[mt] = parse_mf3(lines, mt)
    return out
