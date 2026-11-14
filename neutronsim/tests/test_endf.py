"""Parser/interpolation tests. Fixtures here are SYNTHETIC (invented numbers used only to test code);
they are never presented as nuclear data."""
import math
import pytest
from neutronsim import endf


def f(x: float) -> str:
    if x == 0:
        return " 0.000000+0"
    m, e = f"{x:.6e}".split("e")
    e = int(e)
    return f"{m}{'+' if e >= 0 else '-'}{abs(e)}".rjust(11)


def i(n: int) -> str:
    return f"{n:11d}"


def line(fields, mat=100, mf=3, mt=102, ns=1):
    return "".join(fields).ljust(66) + f"{mat:4d}{mf:2d}{mt:3d}{ns:5d}"


def tab1_lines(xs, ys, laws=((None, 2),), qi=0.0):
    """laws: list of (last_point_index_1based or None, law)."""
    np_ = len(xs)
    laws = [(np_ if b is None else b, l) for b, l in laws]
    out = [line([f(1001.0), f(1.0), i(0), i(0), i(0), i(0)]),
           line([f(qi), f(qi), i(0), i(0), i(len(laws)), i(np_)])]
    pairs = [v for b, l in laws for v in (i(b), i(l))]
    for k in range(0, len(pairs), 6):
        out.append(line(pairs[k:k + 6]))
    data = [f(v) for xy in zip(xs, ys) for v in xy]
    for k in range(0, len(data), 6):
        out.append(line(data[k:k + 6]))
    return out


def tab(xs, ys, laws=((None, 2),), qi=0.0):
    return endf.parse_mf3(tab1_lines(xs, ys, laws, qi), 102)


@pytest.mark.parametrize("text,val", [("1.302700+4", 13027.0), ("-2.500000-3", -2.5e-3),
                                      (" 0.000000+0", 0.0), ("          ", 0.0), ("1.5", 1.5)])
def test_parse_float(text, val):
    assert endf.parse_float(text) == pytest.approx(val)


def test_parse_float_rejects_garbage():
    with pytest.raises(endf.DataError):
        endf.parse_float("abc")


def test_roundtrip_values():
    t = tab([1e-5, 1.0, 10.0], [5.0, 2.0, 1.0])
    assert t.x == pytest.approx([1e-5, 1.0, 10.0]) and t.y == pytest.approx([5.0, 2.0, 1.0])


def test_linlin_midpoint():
    t = tab([1.0, 3.0], [10.0, 20.0])
    assert t(2.0) == pytest.approx(15.0)


def test_histogram():
    t = tab([1.0, 3.0, 5.0], [10.0, 20.0, 30.0], laws=[(None, 1)])
    assert t(2.0) == 10.0 and t(4.0) == 20.0


def test_loglog_power_law_exact():
    xs = [1.0, 100.0]
    t = tab(xs, [1.0, 0.01], laws=[(None, 5)])      # y = 1/x
    assert t(10.0) == pytest.approx(0.1, rel=1e-5)


def test_linlog_and_loglin():
    t = tab([1.0, math.e ** 2], [0.0, 2.0], laws=[(None, 3)])    # y linear in ln x
    assert t(math.e) == pytest.approx(1.0, rel=1e-5)
    t = tab([0.0, 2.0], [1.0, math.e ** 2], laws=[(None, 4)])    # ln y linear in x
    assert t(1.0) == pytest.approx(math.e, rel=1e-5)


def test_law_changes_by_range():
    t = tab([1.0, 2.0, 3.0], [1.0, 3.0, 5.0], laws=[(2, 1), (3, 2)])
    assert t(1.5) == 1.0            # histogram on first interval
    assert t(2.5) == pytest.approx(4.0)  # lin-lin on second


def test_discontinuity_is_right_continuous():
    t = tab([1.0, 2.0, 2.0, 3.0], [1.0, 1.0, 5.0, 5.0])
    assert t(2.0) == 5.0


def test_never_extrapolates():
    t = tab([1.0, 3.0], [10.0, 20.0])
    for e in (0.99, 3.01, 0.0):
        with pytest.raises(endf.DataError):
            t(e)
    assert t(1.0) == 10.0 and t(3.0) == 20.0   # endpoints are in range


def test_log_interp_of_zero_refused():
    t = tab([1.0, 2.0], [0.0, 1.0], laws=[(None, 5)])
    with pytest.raises(endf.DataError):
        t(1.5)


def test_non_monotonic_grid_rejected():
    with pytest.raises(endf.DataError):
        tab([1.0, 3.0, 2.0], [1.0, 1.0, 1.0])


def test_truncated_section_rejected():
    lines = tab1_lines([1.0, 2.0, 3.0, 4.0], [1.0] * 4)[:-1]
    with pytest.raises(endf.DataError):
        endf.parse_mf3(lines, 102)
