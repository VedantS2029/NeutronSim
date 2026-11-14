"""Reaction-channel definitions (ENDF-6 MT numbers).

Names follow the ENDF-6 Formats Manual. Only MTs this program can name with
confidence are listed; any other MT is shown as 'MT<n>' rather than guessed.
"""
from __future__ import annotations

NAMES: dict[int, str] = {
    1: "(n,total)", 2: "(n,n) elastic scattering", 3: "(n,nonelastic)",
    4: "(n,n') inelastic scattering (sum of levels)", 5: "(n,anything)",
    11: "(n,2nd)", 16: "(n,2n)", 17: "(n,3n)", 18: "(n,fission) total",
    22: "(n,n+alpha)", 23: "(n,n+3alpha)", 24: "(n,2n+alpha)", 25: "(n,3n+alpha)",
    28: "(n,n+p)", 29: "(n,n+2alpha)", 30: "(n,2n+2alpha)", 32: "(n,n+d)",
    33: "(n,n+t)", 34: "(n,n+3He)", 35: "(n,n+d+2alpha)", 36: "(n,n+t+2alpha)",
    37: "(n,4n)", 41: "(n,2n+p)", 42: "(n,3n+p)", 44: "(n,n+2p)", 45: "(n,n+p+alpha)",
    102: "(n,gamma) radiative capture", 103: "(n,p)", 104: "(n,d)", 105: "(n,t)",
    106: "(n,3He)", 107: "(n,alpha)", 108: "(n,2alpha)", 109: "(n,3alpha)",
    111: "(n,2p)", 112: "(n,p+alpha)", 113: "(n,t+2alpha)", 114: "(n,d+2alpha)",
    115: "(n,p+d)", 116: "(n,p+t)", 117: "(n,d+alpha)",
}

# Disjoint members of the ENDF non-elastic summation rule (MT3 = sum of these).
_SUMMATION = (
    {4, 5, 11, 16, 17, 18, 28, 29, 30, 37, 41, 42, 44, 45}
    | set(range(22, 26)) | set(range(32, 38))
    | set(range(102, 118))
    | set(range(152, 155)) | set(range(156, 182)) | set(range(183, 201))
)


def is_summation_member(mt: int) -> bool:
    return mt in _SUMMATION


def name(mt: int) -> str:
    return NAMES.get(mt, f"MT{mt}")


def kind(mt: int) -> str:
    """Physical family of a channel (for explanations only)."""
    if mt == 2:
        return "scattering (elastic)"
    if mt in (4, 5):
        return "scattering (inelastic / other)"
    if mt == 18:
        return "fission"
    if mt == 102:
        return "capture"
    if 103 <= mt <= 117:
        return "charged-particle emission"
    if mt in (16, 17, 37) or 22 <= mt <= 45:
        return "other neutron-induced reaction"
    return "other"
