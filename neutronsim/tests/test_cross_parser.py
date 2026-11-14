"""Independent check: this project's reader vs the openmc-dev 'endf' package on the same files."""
import gzip
import tempfile
import pytest

endf_pkg = pytest.importorskip("endf")
from neutronsim import core, endf  # noqa: E402


@pytest.mark.parametrize("iso", ["B-10", "Al-27"])
def test_tables_identical_to_endf_package(iso):
    info = core.json.loads((core.DATA_DIR / "MANIFEST.json").read_text())["isotopes"][iso]
    path = core.DATA_DIR / info["pendf_file"]
    with tempfile.NamedTemporaryFile("w", suffix=".endf", delete=False) as tmp:
        tmp.write(gzip.open(path, "rt").read())
    mat = endf_pkg.Material(tmp.name)
    mine = endf.read_mf3(path)
    assert set(mine) == {mt for (mf, mt) in mat.section_data if mf == 3}
    for mt, t in mine.items():
        ref = mat.section_data[3, mt]["sigma"]
        assert list(ref.x) == pytest.approx(t.x, rel=1e-12)
        assert list(ref.y) == pytest.approx(t.y, rel=1e-12)
