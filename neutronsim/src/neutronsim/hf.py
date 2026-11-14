"""Optical model + statistical (Hauser-Feshbach type) decay of the compound nucleus, stdlib only.

This fills the 1 keV - 20 MeV range that nphys.py used to refuse. It is a SIMPLIFIED, honestly labelled model, not a replacement
for TALYS / EMPIRE. What is computed, and from which law:

  1. Optical model. Neutron + nucleus in a complex Woods-Saxon well (real volume, imaginary volume + surface, Thomas spin-orbit; S_lj for j = l +- 1/2), global
     parameters of the Koning-Delaroche (2003) neutron set (typed from memory of the published fit, NOT re-checked against the
     paper in this sandbox; dispersive terms are left out). The radial Schroedinger equation is integrated with
     Numerov's method in complex arithmetic and matched to Riccati-Hankel functions, giving the S-matrix S_l and the
     transmission coefficient T_l = 1 - |S_l|^2 for every partial wave l.
        sigma_CN    = (pi/k^2) sum (2l+1) T_l             compound-nucleus formation (= reaction cross section)
        sigma_shape = (pi/k^2) sum (2l+1) |1 - S_l|^2     shape (direct) elastic
        sigma_tot   = (2 pi/k^2) sum (2l+1) (1 - Re S_l)
  2. Outgoing charged particles (p, alpha): transmission through the Coulomb + centrifugal barrier at the nuclear surface
     (WKB below the barrier top, Hill-Wheeler form above it); the nuclear interior is taken as perfectly absorbing.
     This tends to OVERestimate charged-particle emission (no preformation factor): see the validation.
  3. Levels. Low-lying discrete levels are ESTIMATED by nucleus type (see discrete_levels), the continuum above is the
     back-shifted Fermi-gas density already used in nphys. Without real level schemes the inelastic cross section near
     threshold is only as good as these estimates.
  4. Decay. Weisskopf-Ewing limit of Hauser-Feshbach: the weight of an exit channel x from a compound state of excitation U is
        N_x = g_x [ sum_discrete tau_x(eps_i) + integral rho(U') tau_x(eps) dU' ],   tau_x(eps) = sum_l (2l+1) T_l(eps)
     and gamma emission uses the same Brink-Axel / TRK strength as nphys (N_gamma = 2 pi Gamma_gamma / D). Branching is
     N_x / sum N. After a particle leaves, the residual nucleus decays again (n, p, alpha, gamma compete) up to any depth
     the energy allows, so (n,n'), (n,2n), (n,3n), (n,np), (n,n alpha) ... all come out of one tree.
  NOT included (each is a known simplification): conservation of total angular momentum and parity between entrance and exit
  channels (that is what 'Weisskopf-Ewing' means), width-fluctuation corrections (Moldauer / HRTW), pre-equilibrium and direct
  inelastic scattering to collective levels, fission, and neutron-proton / alpha spin-orbit terms. Multi-neutron channels
  therefore come from evaporation only, and direct (n,n') on rotor bands is missed.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from functools import lru_cache

from . import nphys, physics

HBARC = 197.3269804           # MeV fm
AMU = 931.49410242            # MeV
E2 = 1.43996454               # MeV fm, e^2/(4 pi eps0)
HF_MAX_E_eV = 20e6            # beyond this: pre-equilibrium dominates, not modelled
H_STEP = 0.2                  # fm, Numerov step
DU = 0.125                    # MeV, excitation-energy bin of the continuum (converged to ~10-20 % on (n,3n): 0.5 MeV bins moved it by a factor 2)
R_BARRIER = 1.3               # fm, charged-particle interaction radius parameter: R_b = 1.3 (A_res^(1/3) + A_x^(1/3))
G_WEIGHT = {"n": 1.0, "p": 1.0, "a": 0.5}      # (2s+1) factors relative to the neutron convention of nphys (Gamma_n/D = T0/2pi)
SPECIES = {"n": (0, 1), "p": (1, 1), "a": (2, 4)}          # (Z, A)
MAX_L = 40
MAX_EMIT = 4                  # at most this many particles are emitted in one decay chain (deeper chains need > 20 MeV: not modelled)
PRUNE = 1e-7                  # exit channels weighing less than this fraction of the total are dropped (keeps the tree finite)


# --------------------------------------------------------------------------- optical model (neutrons)
def kd_potential(a: int, z: int, e_mev: float) -> dict:
    """Koning-Delaroche-type global neutron potential parameters at laboratory energy e_mev (see module docstring caveat)."""
    n = a - z
    ef = -11.2814 + 0.02646 * a
    de = e_mev - ef
    v1 = 59.30 - 21.0 * (n - z) / a - 0.024 * a
    v2 = 7.228e-3 - 1.48e-6 * a
    v3 = 1.994e-5 - 2.0e-8 * a
    v4 = 7e-9
    w1 = 12.195 + 0.0167 * a
    w2 = 73.55 + 0.0795 * a
    d1 = 16.0 - 16.0 * (n - z) / a
    d2 = 0.0180 + 3.802e-3 / (1 + math.exp((a - 156) / 8))
    d3 = 11.5
    return dict(
        vv=v1 * (1 - v2 * de + v3 * de ** 2 - v4 * de ** 3),
        wv=w1 * de ** 2 / (de ** 2 + w2 ** 2),
        wd=d1 * de ** 2 / (de ** 2 + d3 ** 2) * math.exp(-d2 * de),
        vso=(5.922 + 0.0030 * a) * math.exp(-0.0040 * de), wso=-3.1 * de ** 2 / (de ** 2 + 160.0 ** 2),
        rso=(1.1854 - 0.647 * a ** (-1 / 3)) * a ** (1 / 3), aso=0.59,
        rv=(1.3039 - 0.4054 * a ** (-1 / 3)) * a ** (1 / 3), av=0.6778 - 1.487e-4 * a,
        rd=(1.3424 - 0.01585 * a ** (1 / 3)) * a ** (1 / 3), ad=0.5446 - 1.656e-4 * a)


def _kinematics(a: int, e_lab_mev: float) -> tuple[float, float]:
    """(k in 1/fm, 2 mu / hbar^2 in 1/(MeV fm^2)) for a neutron on mass number a, non-relativistic centre of mass."""
    mt = a * AMU
    mu = physics.M_N * mt / (physics.M_N + mt)
    e_cm = e_lab_mev * mt / (physics.M_N + mt)
    return math.sqrt(2 * mu * e_cm) / HBARC, 2 * mu / HBARC ** 2


def _bessel(l: int, x: float) -> tuple[float, float, float, float]:
    """Riccati functions F = x j_l, G = -x y_l (F ~ sin(x - l pi/2), G ~ cos(x - l pi/2)) and their derivatives dF/dx, dG/dx."""
    def f_series(ll: int) -> float:
        term = x ** (ll + 1)
        for i in range(1, 2 * ll + 2, 2):
            term /= i
        s, t = 0.0, term
        for kk in range(60):
            s += t
            t *= -x * x / (2 * (kk + 1) * (2 * ll + 2 * kk + 3))
            if abs(t) < 1e-17 * abs(s) + 1e-300:
                break
        return s

    big_f, big_g = [math.sin(x)], [math.cos(x)]
    if l >= 1:
        big_f.append(math.sin(x) / x - math.cos(x))
        big_g.append(math.cos(x) / x + math.sin(x))
    for ll in range(1, l):
        big_g.append((2 * ll + 1) / x * big_g[ll] - big_g[ll - 1])
        if x > ll + 1:
            big_f.append((2 * ll + 1) / x * big_f[ll] - big_f[ll - 1])
    if x <= l + 1:          # upward recursion of F is unstable for x < l: use the power series instead
        big_f = [f_series(ll) for ll in range(l + 1)]
    f, g = big_f[l], big_g[l]
    if l == 0:
        return f, math.cos(x), g, -math.sin(x)
    return f, big_f[l - 1] - l / x * f, g, big_g[l - 1] - l / x * g


def _s_matrix_l(a: int, z: int, e_mev: float, l: int, pot: dict, k: float, c: float, sigl: float = 0.0) -> complex:
    """S_l for total angular momentum j = l +- 1/2 through sigl = sigma.l = l (j = l+1/2) or -(l+1) (j = l-1/2); 0 gives the spin-independent part only."""
    h = H_STEP
    r_max = max(pot["rv"] + 10 * pot["av"], pot["rd"] + 10 * pot["ad"], 12.0)
    n_steps = int(r_max / h) + 3
    ll1 = l * (l + 1)
    fv = [0j] * (n_steps + 1)
    for i in range(1, n_steps + 1):
        r = i * h
        xv = (r - pot["rv"]) / pot["av"]
        xd = (r - pot["rd"]) / pot["ad"]
        ev = math.exp(min(xv, 700))
        ed = math.exp(min(xd, 700))
        wsv = 1 / (1 + ev)
        surf = 4 * ed / (1 + ed) ** 2
        u = -pot["vv"] * wsv - 1j * (pot["wv"] * wsv + pot["wd"] * surf)
        if sigl:                         # Thomas spin-orbit: (Vso + i Wso) (hbar/m_pi c)^2 (1/r) df/dr  sigma.l
            xs = (r - pot["rso"]) / pot["aso"]
            es = math.exp(min(xs, 700))
            dfdr = -es / (1 + es) ** 2 / pot["aso"]
            u += (pot["vso"] + 1j * pot["wso"]) * 2.0 * dfdr / r * sigl
        fv[i] = ll1 / (r * r) + c * u - k * k
    u0, u1 = 0j, complex(h ** (l + 1))
    us = [u0, u1, complex((2 * h) ** (l + 1))]
    h12 = h * h / 12
    for i in range(2, n_steps):
        nxt = (2 * (1 + 5 * h12 * fv[i]) * us[i] - (1 - h12 * fv[i - 1]) * us[i - 1]) / (1 - h12 * fv[i + 1])
        us.append(nxt)
        if abs(nxt) > 1e150:             # keep numbers finite; only the log derivative matters
            us = [v * 1e-150 for v in us]
    m = int(r_max / h)
    du = (-us[m + 2] + 8 * us[m + 1] - 8 * us[m - 1] + us[m - 2]) / (12 * h)
    r_m = m * h
    log_d = r_m * du / us[m]
    x = k * r_m
    f, fp, g, gp = _bessel(l, x)
    o, op = complex(g, f), complex(gp, fp)           # outgoing  G + iF
    i_, ip = complex(g, -f), complex(gp, -fp)        # incoming  G - iF
    return (x * ip - log_d * i_) / (x * op - log_d * o)


@lru_cache(maxsize=4096)
def neutron_smatrix(a: int, z: int, e_lab_eV: float) -> tuple[tuple[tuple[complex, complex], ...], float]:
    """((S_l+, S_l-) for l = 0, 1, ...; k): partial-wave S-matrix of a neutron of e_lab_eV on (z, a), for j = l + 1/2 and j = l - 1/2
    (for l = 0 both entries are the j = 1/2 value). a = target mass number."""
    e_mev = e_lab_eV * 1e-6
    pot = kd_potential(a, z, e_mev)
    k, c = _kinematics(a, e_mev)
    kr = k * max(pot["rv"], pot["rd"])
    out = []
    for l in range(MAX_L):
        sp = _s_matrix_l(a, z, e_mev, l, pot, k, c, l)
        sm = _s_matrix_l(a, z, e_mev, l, pot, k, c, -(l + 1)) if l else sp
        out.append((sp, sm))
        if l > kr + 1 and 1 - abs(sp) ** 2 < 1e-9 and 1 - abs(sm) ** 2 < 1e-9:
            break
    return tuple(out), k


def _weights(l: int) -> tuple[float, float]:
    """(2j+1)/2 for j = l+1/2 and l-1/2: they add up to (2l+1), so spin-summed formulas keep the sum over (2l+1) T_l form."""
    return (l + 1.0, float(l))


def transmission(a: int, z: int, e_lab_eV: float) -> list[float]:
    s, _k = neutron_smatrix(a, z, e_lab_eV)
    out = []
    for l, (sp, sm) in enumerate(s):
        wp, wm = _weights(l)
        t = (wp * (1 - abs(sp) ** 2) + wm * (1 - abs(sm) ** 2)) / (2 * l + 1)       # j-averaged T_l
        out.append(min(1.0, max(0.0, t)))
    return out


def optical_cross_sections(a: int, z: int, e_lab_eV: float) -> dict[str, float]:
    """Neutron on (z, a): reaction (= compound-nucleus formation), shape elastic and total cross sections in barn."""
    s, k = neutron_smatrix(a, z, e_lab_eV)
    f = math.pi / k ** 2 / 100.0
    react = shape = total = 0.0
    for l, pair in enumerate(s):
        for w, v in zip(_weights(l), pair):
            react += w * (1 - abs(v) ** 2)
            shape += w * abs(1 - v) ** 2
            total += 2 * w * (1 - v.real)
    react, shape, total = f * react, f * shape, f * total
    return {"reaction": max(react, 0.0), "shape_elastic": shape, "total": total}


# --------------------------------------------------------------------------- transmission tables for emitted particles
GRID_RATIO = 1.2
_E_GRID = [1e-3 * GRID_RATIO ** i for i in range(int(math.log(45.0 / 1e-3) / math.log(GRID_RATIO)) + 2)]       # MeV, 1 keV .. ~45 MeV


def _anchor(a: int, z: int) -> tuple[int, int]:
    """Neutron optical-model tables are computed for the exact residual nucleus (an earlier version shared tables between neighbours: faster, less exact)."""
    return a, z


def _tau_n_table(a_res: int, z_res: int) -> tuple[float, ...]:
    return _tau_n_table_exact(*_anchor(a_res, z_res))


@lru_cache(maxsize=None)
def _tau_n_table_exact(a_res: int, z_res: int) -> tuple[float, ...]:
    out = []
    for e in _E_GRID:
        out.append(sum((2 * l + 1) * t for l, t in enumerate(transmission(a_res, z_res, e * 1e6))))
    return tuple(out)


def _interp_log(table: tuple[float, ...], e: float) -> float:
    if e <= _E_GRID[0]:
        return table[0] * math.sqrt(max(e, 0.0) / _E_GRID[0])        # s-wave: T ~ sqrt(E) (neutrons), also keeps charged ~0
    if e >= _E_GRID[-1]:
        return table[-1]
    x = math.log(e / _E_GRID[0]) / math.log(GRID_RATIO)
    i = int(x)
    t = x - i
    lo, hi = table[i], table[i + 1]
    if lo > 0 and hi > 0:                       # tau rises by orders of magnitude through a Coulomb barrier: interpolate its logarithm
        return math.exp(math.log(lo) * (1 - t) + math.log(hi) * t)
    return lo * (1 - t) + hi * t


def barrier_transmission(zx: int, ax: int, z_res: int, a_res: int, eps: float, l: int) -> float:
    """Transmission of a charged particle (zx, ax) of kinetic energy eps (MeV) leaving a nucleus (z_res, a_res) with angular momentum l."""
    if eps <= 0:
        return 0.0
    mu = ax * a_res / (ax + a_res) * AMU
    rb = R_BARRIER * (a_res ** (1 / 3) + ax ** (1 / 3))
    zz = zx * z_res * E2
    cen = HBARC ** 2 * l * (l + 1) / (2 * mu)
    v_top = zz / rb + cen / rb ** 2
    if eps >= v_top:
        curv = abs(2 * zz / rb ** 3 + 6 * cen / rb ** 4)
        hw = HBARC * math.sqrt(curv / mu) if curv > 0 else 1.0
        return 1 / (1 + math.exp(min(-2 * math.pi * (eps - v_top) / max(hw, 1e-6), 700)))
    # outer turning point: eps r^2 - zz r - cen = 0
    r2 = (zz + math.sqrt(zz * zz + 4 * eps * cen)) / (2 * eps)
    n = 40
    s = 0.0
    for i in range(n):
        th = (i + 0.5) * (math.pi / 2) / n
        r = rb + (r2 - rb) * math.sin(th) ** 2
        v = zz / r + cen / r ** 2
        s += math.sqrt(max(v - eps, 0.0) * 2 * mu) / HBARC * 2 * (r2 - rb) * math.sin(th) * math.cos(th) * (math.pi / 2) / n
    return 1 / (1 + math.exp(min(2 * s, 700)))


@lru_cache(maxsize=None)
def _tau_charged_table(zx: int, ax: int, z_res: int, a_res: int) -> tuple[float, ...]:
    out = []
    for e in _E_GRID:
        tot = 0.0
        for l in range(MAX_L):
            t = barrier_transmission(zx, ax, z_res, a_res, e, l)
            tot += (2 * l + 1) * t
            if l > 2 and t < 1e-9:
                break
        out.append(tot)
    return tuple(out)


def tau(x: str, z_res: int, a_res: int, eps: float) -> float:
    """sum_l (2l+1) T_l for particle x ('n', 'p', 'a') of kinetic energy eps (MeV) leaving a residual nucleus (z_res, a_res)."""
    if eps <= 0:
        return 0.0
    if x == "n":              # eps is the centre-of-mass energy; the optical potential is defined at the laboratory energy of the neutron
        return _interp_log(_tau_n_table(a_res, z_res), eps * (a_res + 1) / a_res)
    zx, ax = SPECIES[x]
    return _interp_log(_tau_charged_table(zx, ax, z_res, a_res), eps)


# --------------------------------------------------------------------------- levels
def _valence(z: int, n: int) -> float:
    """Casten-style valence product P = Np Nn / (Np + Nn); 0 at a closed shell."""
    def dist(k):
        return min(abs(k - m) for m in nphys.MAGIC if m >= 8 or k <= 8) if k > 0 else 0
    np_, nn = dist(z), dist(n)
    return np_ * nn / (np_ + nn) if np_ + nn else 0.0


def first_excited_mev(z: int, n: int) -> float:
    """Rough energy of the first collective state of an even-even nucleus (2+): empirical decrease with the valence product."""
    a = z + n
    if z in nphys.MAGIC and n in nphys.MAGIC:
        return 3.0
    e = 1.5 * math.exp(-0.35 * _valence(z, n))
    if a < 40:
        e *= (40 / a) ** 1.5
    if z in nphys.MAGIC or n in nphys.MAGIC:
        e = max(e, 1.0)
    return min(max(e, 0.04), 4.0)


@lru_cache(maxsize=None)
def discrete_levels(z: int, n: int) -> tuple[float, ...]:
    """ESTIMATED excitation energies (MeV) of the lowest levels, ground state first. Rule by nucleus type:
    even-even: 2+ from first_excited_mev; rotor (E2 < 0.3) 4+, 6+, 8+ at 3.3, 7, 12 E2; vibrator two-phonon triplet at 2 E2, 3 E2 and 3.3 E2;
    odd-A: single-particle-like levels at 0.6, 1.3, 2.0, 2.7 times E2; odd-odd: dense, at 0.25, 0.5, 0.8, 1.1, 1.5 times E2.
    No spins or parities are assigned; every level counts once."""
    e2 = first_excited_mev(z, n)
    if z % 2 == 0 and n % 2 == 0:
        mult = (0, 1, 3.3, 7, 12) if e2 < 0.3 else (0, 1, 2, 2, 2, 3, 3.3)
    elif (z + n) % 2 == 1:
        mult = (0, 0.6, 1.3, 2.0, 2.7)
    else:
        mult = (0, 0.25, 0.5, 0.8, 1.1, 1.5)
    return tuple(m * e2 for m in mult)


def _du(z: int, n: int) -> float:
    return DU


def continuum_start(z: int, n: int) -> float:
    lv = discrete_levels(z, n)
    return max(lv[-1] + 0.25 * lv[1], 0.5) if len(lv) > 1 else 0.5


@lru_cache(maxsize=None)
def _rho_bins(z: int, n: int, jmax: int) -> tuple[float, ...]:
    """Level density at the centres of the continuum bins of nucleus (z, n): U_j = E_c + (j + 1/2) DU."""
    ec, a = continuum_start(z, n), z + n
    return tuple(nphys._rho(a, z, n, ec + (j + 0.5) * _du(z, n))[0] for j in range(jmax))


def _binding(z: int, n: int, x: str) -> float:
    """Energy needed to remove particle x from nucleus (z, n): M(res) + M(x) - M(parent), model masses (MeV)."""
    zx, ax = SPECIES[x]
    zr, nr = z - zx, n - (ax - zx)
    mx = {"n": physics.M_N, "p": physics.M_P + physics.M_E, "a": physics.M_HE4}[x]
    return physics.atomic_mass(zr, nr) + mx - physics.atomic_mass(z, n)


def _valid(z: int, n: int) -> bool:
    return z >= 1 and n >= 0 and z + n >= 1


def gamma_weight(z: int, n: int, u: float) -> float:
    """N_gamma = 2 pi Gamma_gamma / D at excitation u of nucleus (z, n): the nphys Brink-Axel/TRK estimate generalised to any u."""
    if u <= 0.1 or z + n < 4:
        return 0.0
    a = z + n
    steps = 48
    tot = 0.0
    for i in range(1, steps):
        eg = u * i / steps
        tot += nphys._f_e1(eg, a, z, n) * eg ** 3 * nphys._rho(a, z, n, u - eg)[0] * (u / steps)
    return nphys.SPIN_FACTOR * tot


# --------------------------------------------------------------------------- the decay tree
Outcome = tuple      # (n_neutrons, n_protons, n_alphas) emitted in total


_INC = {"n": (1, 0, 0), "p": (0, 1, 0), "a": (0, 0, 1)}


def _add(counts: Outcome, x: str) -> Outcome:
    d = _INC[x]
    return (counts[0] + d[0], counts[1] + d[1], counts[2] + d[2])


def _channels(z: int, n: int, u: float, root: bool = False):
    """Exit channels from nucleus (z, n) at excitation u: list of (weight, x, child) with child = None (terminal: the residual
    ends in a discrete level and de-excites by gamma), ('el',) for the root's neutron to the target ground state, or
    (z_res, n_res, bin) for a residual in the continuum."""
    out = [(gamma_weight(z, n, u), None, None)]
    for x in "npa":
        zx, ax = SPECIES[x]
        zr, nr = z - zx, n - (ax - zx)
        if not _valid(zr, nr) or nr < 0:
            continue
        ek = u - _binding(z, n, x)
        if ek <= 1e-3:
            continue
        ar = zr + nr
        g = G_WEIGHT[x]
        for i, e_lev in enumerate(discrete_levels(zr, nr)):
            if ek - e_lev > 1e-3:
                w = g * tau(x, zr, ar, ek - e_lev)
                if w > 0:
                    out.append((w, x, ("el",) if (root and x == "n" and i == 0) else None))
        ec = continuum_start(zr, nr)
        if ek > ec:
            du = _du(zr, nr)
            nb = int((ek - ec) / du) + 1
            rho = _rho_bins(zr, nr, nb + 1)
            for j in range(nb):
                lo = ec + j * du
                width = min(du, ek - lo)
                if width <= 0:
                    continue
                u_res = lo + width / 2
                w = g * tau(x, zr, ar, ek - u_res) * rho[j] * width
                if w > 0:
                    out.append((w, x, (zr, nr, j)))
    return out


@lru_cache(maxsize=None)
def _node(z: int, n: int, j: int, depth: int) -> tuple[tuple[Outcome, float], ...]:
    if depth <= 0:                       # emission budget used up: the rest is gamma de-excitation
        return (((0, 0, 0), 1.0),)
    u = continuum_start(z, n) + (j + 0.5) * _du(z, n)
    return _combine(_channels(z, n, u), depth)


def _combine(chs, depth: int) -> tuple[tuple[Outcome, float], ...]:
    tot = sum(w for w, _x, _c in chs)
    res: dict[Outcome, float] = {}
    if tot <= 0:
        return (((0, 0, 0), 1.0),)
    for w, x, child in chs:
        p = w / tot
        if p < PRUNE:
            continue
        if x is None:
            res[(0, 0, 0)] = res.get((0, 0, 0), 0.0) + p
        elif child is None:
            key = _add((0, 0, 0), x)
            res[key] = res.get(key, 0.0) + p
        else:
            for oc, q in _node(*child, depth - 1):
                key = _add(oc, x)
                res[key] = res.get(key, 0.0) + p * q
    norm = sum(res.values())
    return tuple((k, v / norm) for k, v in res.items()) if norm > 0 else (((0, 0, 0), 1.0),)


@dataclass
class HFResult:
    energy_eV: float
    sigma_cn_b: float
    sigma_shape_b: float
    sigma_total_b: float
    excitation_mev: float
    branches: dict = field(default_factory=dict)       # outcome key -> probability from the compound nucleus
    compound_elastic: float = 0.0                       # probability of neutron back to the target ground state

    def sigma(self, key) -> float:
        return self.sigma_cn_b * self.branches.get(key, 0.0)

    @property
    def elastic_b(self) -> float:
        return self.sigma_shape_b + self.sigma_cn_b * self.compound_elastic


# outcome tuple -> ENDF MT. The visualizer (viz.EMISSIONS) knows the emitted particles of every MT used here.
MT_OF = {(0, 0, 0): 102, (1, 0, 0): 4, (2, 0, 0): 16, (3, 0, 0): 17, (4, 0, 0): 37, (0, 1, 0): 103, (0, 0, 1): 107,
         (1, 1, 0): 28, (1, 0, 1): 22, (2, 1, 0): 41, (2, 0, 1): 24, (0, 2, 0): 111, (0, 1, 1): 112}
LABEL_OF = {102: "capture (n,gamma)", 4: "inelastic (n,n')", 16: "(n,2n)", 17: "(n,3n)", 37: "(n,4n)", 103: "(n,p)", 107: "(n,alpha)",
            28: "(n,n+p)", 22: "(n,n+alpha)", 41: "(n,2n+p)", 24: "(n,2n+alpha)", 111: "(n,2p)", 112: "(n,p+alpha)"}


def available(z: int, n: int) -> str | None:
    """None if the model can build a channel set for target (z, n), else why not."""
    if nphys.gamma_over_spacing(z, n) is None:
        return "the compound nucleus is too light or too weakly bound for the model"
    return None


@lru_cache(maxsize=2048)
def decay(z: int, n: int, e_lab_eV: float) -> HFResult:
    """Full forecast of neutron + (z, n) at e_lab_eV: optical-model sigma_CN, shape elastic, and the branching of the compound nucleus."""
    why = available(z, n)
    if why:
        raise ValueError(why)
    if e_lab_eV > HF_MAX_E_eV:
        raise ValueError("above 20 MeV pre-equilibrium emission dominates and is not modelled")
    a = z + n
    oc = optical_cross_sections(a, z, e_lab_eV)
    k_e = e_lab_eV * 1e-6 * a * AMU / (physics.M_N + a * AMU)
    sn = _binding(z, n + 1, "n")
    u0 = sn + k_e
    chs = _channels(z, n + 1, u0, root=True)
    tot = sum(w for w, _x, _c in chs)
    br: dict[Outcome, float] = {}
    ce = 0.0
    for w, x, child in chs:
        p = w / tot if tot > 0 else 0.0
        if p < PRUNE:
            continue
        if x is None:
            br[(0, 0, 0)] = br.get((0, 0, 0), 0.0) + p
        elif child == ("el",):
            ce += p
        elif child is None:
            key = _add((0, 0, 0), x)
            br[key] = br.get(key, 0.0) + p
        else:
            for ocm, q in _node(*child, MAX_EMIT - 1):
                key = _add(ocm, x)
                br[key] = br.get(key, 0.0) + p * q
    return HFResult(e_lab_eV, oc["reaction"], oc["shape_elastic"], oc["total"], u0, br, ce)


def channel_sigmas(res: HFResult) -> dict[int, float]:
    """MT -> cross section (barn); outcomes with no ENDF MT in MT_OF are summed under MT 5 ('anything else')."""
    out: dict[int, float] = {}
    for key, p in res.branches.items():
        mt = MT_OF.get(key, 5)
        out[mt] = out.get(mt, 0.0) + res.sigma_cn_b * p
    return out


def thresholds_mev(z: int, n: int) -> dict[str, float]:
    """Laboratory neutron energy (MeV) below which each channel is energetically closed for target (z, n), from the model masses
    (same accuracy as physics.py: errors of a few MeV are possible). (n,n') uses the estimated first excited level."""
    a = z + n
    fac = (a + 1) / a
    sn_cn = _binding(z, n + 1, "n")
    out = {"(n,n') first level": (discrete_levels(z, n)[1] if len(discrete_levels(z, n)) > 1 else 0.5) * fac}
    if n >= 1:
        out["(n,2n)"] = max(0.0, _binding(z, n, "n")) * fac
    for x, name in (("p", "(n,p)"), ("a", "(n,alpha)")):
        zx, ax = SPECIES[x]
        if z - zx >= 1 and n + 1 - (ax - zx) >= 0:
            out[name] = max(0.0, _binding(z, n + 1, x) - sn_cn) * fac
    return out


# --------------------------------------------------------------------------- check of the estimated first excited level
# Lowest excited state (MeV) of well-known nuclides, typed from memory of standard level schemes (good to a few percent). Even-even: 2+ ; odd-A: first excited state.
LEVEL_REFERENCE = {
    "even-even": {"Mg-24": 1.369, "Si-28": 1.779, "S-32": 2.230, "Ca-40": 3.904, "Ti-48": 0.984, "Cr-52": 1.434, "Fe-56": 0.847, "Ni-58": 1.454,
                  "Zn-64": 0.992, "Ge-72": 0.834, "Zr-90": 2.186, "Mo-98": 0.787, "Pd-106": 0.512, "Cd-114": 0.558, "Sn-120": 1.171, "Te-130": 0.839,
                  "Ba-138": 1.436, "Sm-152": 0.122, "Gd-156": 0.089, "Er-166": 0.081, "W-184": 0.111, "Os-190": 0.187, "Pt-196": 0.356, "Hg-202": 0.439,
                  "Pb-208": 2.614, "Th-232": 0.049, "U-238": 0.045, "Pu-240": 0.043, "Ne-20": 1.634},
    "odd-A": {"Na-23": 0.440, "Al-27": 0.844, "Sc-45": 0.012, "V-51": 0.320, "Mn-55": 0.126, "Fe-57": 0.014, "Co-59": 1.099, "Cu-63": 0.670,
              "Nb-93": 0.031, "Ag-109": 0.088, "I-127": 0.058, "Au-197": 0.077, "Pb-207": 0.570, "Bi-209": 0.896, "U-235": 0.013},
}


def level_validation() -> tuple[list[str], dict[str, float]]:
    """rms of log10(estimated / real) for the first excited level, per nucleus type. The odd-A scatter is large because real odd-A first levels
    depend on single-particle structure (bandheads and isomers at a few keV next to levels near 1 MeV) that no smooth rule captures; a Fermi-gas
    placement of the levels was tried and was no better (0.83 dex)."""
    from .nuclide import Nuclide
    out, rms = ["Estimated first excited level vs real (log10 ratio, rms):"], {}
    for kind, d in LEVEL_REFERENCE.items():
        lg = []
        for k, v in d.items():
            nu = Nuclide.parse(k)
            lv = discrete_levels(nu.Z, nu.N)
            lg.append(math.log10(lv[1] / v))
        rms[kind] = (sum(x * x for x in lg) / len(lg)) ** 0.5
        out.append(f"  {kind:<10} {rms[kind]:.2f} dex over {len(lg)} nuclides (factor {10 ** rms[kind]:.1f})")
    return out, rms


# Laboratory thresholds (MeV) of well-known reactions, typed from memory of standard tables (good to ~0.1 MeV).
THRESHOLD_REFERENCE = {
    "Al-27": {"(n,p)": 1.90, "(n,2n)": 13.5}, "Fe-56": {"(n,p)": 2.97, "(n,2n)": 11.4}, "Ni-58": {"(n,p)": 0.0, "(n,2n)": 12.4},
    "Cu-63": {"(n,p)": 0.73, "(n,2n)": 10.9}, "Au-197": {"(n,2n)": 8.1}, "Pb-208": {"(n,2n)": 7.4}, "Ag-109": {"(n,2n)": 9.3},
    "U-238": {"(n,2n)": 6.2}, "Mg-24": {"(n,p)": 4.9, "(n,2n)": 16.2}, "Zn-64": {"(n,2n)": 12.1},
}


def threshold_validation() -> tuple[list[str], float]:
    """rms error (MeV) of the model-mass thresholds against the reference; the liquid-drop mass formula has no shell effects."""
    from .nuclide import Nuclide
    errs = []
    for k, d in THRESHOLD_REFERENCE.items():
        nu = Nuclide.parse(k)
        th = thresholds_mev(nu.Z, nu.N)
        errs += [th[q] - v for q, v in d.items() if q in th]
    rms = (sum(e * e for e in errs) / len(errs)) ** 0.5
    return [f"Model channel thresholds vs known values: rms error {rms:.2f} MeV, worst {max(errs, key=abs):+.2f} MeV over {len(errs)} thresholds"], rms
