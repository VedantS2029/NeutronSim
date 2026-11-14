import pytest
from neutronsim import core

ISOTOPES = ["B-10", "Al-27", "Ag-109", "U-235"]
_cache: dict[str, core.Dataset] = {}


def get(iso: str) -> core.Dataset:
    if iso not in _cache:
        _cache[iso] = core.Dataset(iso)
    return _cache[iso]


@pytest.fixture(scope="session")
def ds():
    return get
