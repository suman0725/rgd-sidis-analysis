"""
physics.py

Helper functions for CLAS12 DIS-style kinematics.
Naming style: get_<something>.
"""

import numpy as np

from physics_constants import (
    M_PROTON,
    M_NEUTRON,
    M_NUCLEON,
    M_ELECTRON,
    DEG2RAD,
    RAD2DEG,
    M_PION_PLUS, 
)

# -----------------------
# Basic momentum & angles
# -----------------------

def get_p(px, py, pz):
    """
    Return |p| from (px, py, pz).
    """
    px = np.asarray(px)
    py = np.asarray(py)
    pz = np.asarray(pz)
    return np.sqrt(px**2 + py**2 + pz**2)


def get_theta(px, py, pz, degrees=True):
    """
    Return polar angle theta from +z in [0, pi] (or [0, 180] deg).
    """
    px = np.asarray(px)
    py = np.asarray(py)
    pz = np.asarray(pz)

    p = get_p(px, py, pz)
    with np.errstate(divide="ignore", invalid="ignore"):
        costh = np.where(p == 0, np.nan, pz / p)
    costh = np.clip(costh, -1.0, 1.0)
    theta = np.arccos(costh)

    if degrees:
        theta = theta * RAD2DEG
    return theta


def get_phi(px, py, degrees=True):
    """
    Return azimuthal angle phi from (px, py).

    Range: (-pi, pi] or (-180, 180] deg if degrees=True.
    """
    px = np.asarray(px)
    py = np.asarray(py)

    phi = np.arctan2(py, px)  # radians
    if degrees:
        phi = phi * RAD2DEG
    return phi


# -----------------------
# Sector from phi (6 sectors)
# -----------------------

def get_sector(phi_deg):
    """
    Return CLAS12 sector (1..6) from azimuth phi [deg].

    Ranges:
      1: -30  <= phi <  30
      2:  30  <= phi <  90
      3:  90  <= phi < 150
      4:  150 <= phi <= 180  or  -180 <= phi < -150
      5: -150 <= phi < -90
      6: -90  <= phi < -30

    Phi is first wrapped to (-180, 180].
    """
    phi = np.asarray(phi_deg, dtype=float)

    # Wrap to (-180, 180]
    phi_wrapped = ((phi + 180.0) % 360.0) - 180.0

    sector = np.zeros_like(phi_wrapped, dtype=int)

    m1 = (phi_wrapped >= -30.0) & (phi_wrapped <  30.0)
    m2 = (phi_wrapped >=  30.0) & (phi_wrapped <  90.0)
    m3 = (phi_wrapped >=  90.0) & (phi_wrapped < 150.0)
    m4 = (phi_wrapped >= 150.0) | (phi_wrapped < -150.0)
    m5 = (phi_wrapped >= -150.0) & (phi_wrapped < -90.0)
    m6 = (phi_wrapped >= -90.0)  & (phi_wrapped < -30.0)

    sector[m1] = 1
    sector[m2] = 2
    sector[m3] = 3
    sector[m4] = 4
    sector[m5] = 5
    sector[m6] = 6

    return sector


# -----------------------
# Energy & four-vector
# -----------------------

def get_energy(px, py, pz, mass=M_ELECTRON):
    """
    Return energy E = sqrt(p^2 + m^2) [GeV].
    """
    p = get_p(px, py, pz)
    return np.sqrt(p**2 + mass**2)


def get_four_vector(px, py, pz, mass=M_ELECTRON):
    """
    Return four-vector (E, px, py, pz).

    Shape: (..., 4)
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    pz = np.asarray(pz, dtype=float)
    E  = get_energy(px, py, pz, mass=mass)

    E, px, py, pz = np.broadcast_arrays(E, px, py, pz)
    return np.stack([E, px, py, pz], axis=-1)


# -----------------------
# DIS invariants from beam + lepton momentum
# -----------------------
# Inputs: E_beam, px, py, pz of the scattered lepton
# (by default mass_e = M_ELECTRON, target mass = M_NUCLEON)
# -----------------------

def get_Q2(E_beam, px, py, pz, mass_e=M_ELECTRON):
    """
    Return Q^2 [GeV^2] from beam energy and scattered lepton momentum.

    Assumes beam along +z: k = (E_beam, 0, 0, E_beam).
    """
    # scattered lepton
    E_e = get_energy(px, py, pz, mass=mass_e)

    # compute scattering angle via momentum
    theta = get_theta(px, py, pz, degrees=False)

    # textbook relation
    return 4.0 * E_beam * E_e * np.sin(theta / 2.0) ** 2


def get_nu(E_beam, px, py, pz, mass_e=M_ELECTRON):
    """
    Return nu = E_beam - E' [GeV].
    """
    E_e = get_energy(px, py, pz, mass=mass_e)
    return E_beam - E_e


def get_xB(E_beam, px, py, pz, m_target=M_NUCLEON, mass_e=M_ELECTRON):
    """
    Return Bjorken x_B from beam energy and lepton momentum.
    """
    Q2 = get_Q2(E_beam, px, py, pz, mass_e=mass_e)
    nu = get_nu(E_beam, px, py, pz, mass_e=mass_e)
    Q2 = np.asarray(Q2, dtype=float)
    nu = np.asarray(nu, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        xB = Q2 / (2.0 * m_target * np.where(nu == 0.0, np.nan, nu))
    return xB


def get_y(E_beam, px, py, pz, mass_e=M_ELECTRON):
    """
    Return inelasticity y = nu / E_beam.
    """
    nu = get_nu(E_beam, px, py, pz, mass_e=mass_e)
    nu = np.asarray(nu, dtype=float)
    return nu / E_beam


def get_W(E_beam, px, py, pz, m_target=M_NUCLEON, mass_e=M_ELECTRON):
    """
    Return invariant mass W [GeV] of the hadronic system.

    W^2 = m_target^2 + 2 m_target nu - Q^2
    """
    Q2 = get_Q2(E_beam, px, py, pz, mass_e=mass_e)
    nu = get_nu(E_beam, px, py, pz, mass_e=mass_e)

    Q2 = np.asarray(Q2, dtype=float)
    nu = np.asarray(nu, dtype=float)

    W2 = m_target**2 + 2.0 * m_target * nu - Q2
    W2 = np.clip(W2, 0.0, None)
    return np.sqrt(W2)


# -----------------------
# SIDIS helpers: z_h, pT^2, phi_h
# -----------------------


def get_zh(E_beam,
           px_e, py_e, pz_e,
           px_h, py_h, pz_h,
           mass_h=M_PION_PLUS):
    """
    z_h ≈ E_h / nu  in fixed-target lab frame,
    where nu = E_beam - E_e.
    """
    # scattered electron energy
    E_e = get_energy(px_e, py_e, pz_e, mass=M_ELECTRON)
    nu  = E_beam - E_e

    # hadron energy
    px_h = np.asarray(px_h, dtype=float)
    py_h = np.asarray(py_h, dtype=float)
    pz_h = np.asarray(pz_h, dtype=float)

    p_h  = np.sqrt(px_h**2 + py_h**2 + pz_h**2)
    E_h  = np.sqrt(p_h**2 + mass_h**2)

    zh = np.zeros_like(E_h, dtype=float)
    mask = nu != 0.0
    zh[mask] = E_h[mask] / nu[mask]
    return zh

def get_pt2(E_beam,
            px_e, py_e, pz_e,
            px_h, py_h, pz_h):
    """
    pT^2 of hadron wrt q (lab frame), using

        pT^2 = |q × p_h|^2 / |q|^2

    with q_vec = k_in_vec - k_out_vec,
    k_in_vec = (0, 0, E_beam), k_out_vec = (px_e, py_e, pz_e).
    """
    px_e = np.asarray(px_e, dtype=float)
    py_e = np.asarray(py_e, dtype=float)
    pz_e = np.asarray(pz_e, dtype=float)

    px_h = np.asarray(px_h, dtype=float)
    py_h = np.asarray(py_h, dtype=float)
    pz_h = np.asarray(pz_h, dtype=float)

    # 3-vector of q: k_in - k_out
    qx = -px_e
    qy = -py_e
    qz = E_beam - pz_e
    q3 = np.stack([qx, qy, qz], axis=-1)

    # hadron 3-vector
    p3 = np.stack([px_h, py_h, pz_h], axis=-1)

    # |q × p_h|^2 and |q|^2
    cross = np.cross(q3, p3)
    num = np.sum(cross**2, axis=-1)
    den = np.sum(q3**2,   axis=-1)

    pT2 = np.zeros_like(num)
    mask = den > 0.0
    pT2[mask] = num[mask] / den[mask]
    return pT2


def get_phih(E_beam,
              px_e, py_e, pz_e,
              px_h, py_h, pz_h,
              degrees=False):
    """
    you can refer the paper to see the full equation: https://arxiv.org/pdf/hep-ph/0410050 another paper form same group more in detail : https://arxiv.org/pdf/hep-ph/0611265
    Calculates the SIDIS azimuthal angle phi_h (Trento Convention).
    Range: [-180, 180] degrees (if degrees=True).
    """
    # Cast inputs to arrays
    #l(l) + p(P) → l(l) + h(Ph) + X, where l denotes the beam lepton, p the proton target, and h the produced hadron
    # 3-momentum vector for scattered lepton
    px_e = np.asarray(px_e, dtype=float)
    py_e = np.asarray(py_e, dtype=float)
    pz_e = np.asarray(pz_e, dtype=float)

    # 3-momentum vector for scattered hadron
    px_h = np.asarray(px_h, dtype=float)
    py_h = np.asarray(py_h, dtype=float)
    pz_h = np.asarray(pz_h, dtype=float)

    # 1. Define Vectors
    # Virtual Photon q = l - l'  , l for incident and l' for scattered lepton; q represents the virtual photon momentum vector
    # Beam l = (E_beam, 0, 0, pz), where E_beam^2 = pz^2 + m_e^2; m_e^2 is negligible comparing with pz^2; hence pz~ E_beam
    
    qx = -px_e
    qy = -py_e
    qz = E_beam - pz_e
    q3 = np.stack([qx, qy, qz], axis=-1)

    # Beam Vector l
    l3 = np.stack([
        np.zeros_like(px_e),
        np.zeros_like(px_e),
        np.full_like(px_e, E_beam, dtype=float)
    ], axis=-1)

    # Hadron Vector p_h
    ph3 = np.stack([px_h, py_h, pz_h], axis=-1)

    # 2. Normalize q (Critical for ATan2 scaling!)
    qnorm = np.linalg.norm(q3, axis=-1) # qnorm is magnitude |q|
    phi_rad = np.zeros_like(qnorm)

    good = qnorm > 0.0
    if np.any(good):
        qhat = np.zeros_like(q3)
        qhat[good] = (q3[good].T / qnorm[good]).T

        # 3. Cross Products
        # v_lep direction ~ (q x k)
        cross_ql  = np.cross(qhat, l3)
        # v_had direction ~ (q x ph)
        cross_qph = np.cross(qhat, ph3)
        # For numerator term
        cross_lph = np.cross(l3, ph3)

        # 4. Calculate Terms
        # Y-term (Sine-like): (l x ph) . qhat
        num = np.einsum("...i,...i->...", cross_lph, qhat)
        
        # X-term (Cosine-like): (qhat x l) . (qhat x ph)
        den = np.einsum("...i,...i->...", cross_ql,  cross_qph)

        # 5. ATan2 (Range is -pi to +pi)
        phi_rad[good] = np.arctan2(num[good], den[good])

    if degrees:
        # Shift range from [-180, 180] to [0, 360]
        return (phi_rad * RAD2DEG) + 180.0
        
    return phi_rad
