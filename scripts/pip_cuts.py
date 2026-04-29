import numpy as np
from common_cuts import is_fd_pip

# ---------- 1) Vz Windows (Positive Pions) ----------
# OLD values (some incorrect vs note):
# VZ_WINDOWS_OB = {
#     "LD2": (-20.0000, 5.0000),   # lower bound was wrong (-20 vs -15)
#     "CxC": (-10.5319, 5.0000),   # slightly off
#     "Cu":  (-9.76024, -5.1676),  # close
#     "Sn":  (-4.6413,  5.0000),   # upper bound was wrong (5.0 vs 0.1)
# }
# NEW — Table 6 (π+ OB) from RGD_CommonNote2
VZ_WINDOWS_OB = {
    "LD2": (-15.0, 5.0),
    "CxC": (-10.1, 5.0),
    "Cu":  (-9.8,  -5.2),
    "Sn":  (-5.2,   0.1),
}

VZ_WINDOWS_IB = {
    "LD2": (-15.0000, 5.0000),
    "CxC": (-10.28,   5.0000),
    "Cu":  (-10.69,  -6.50),
    "Sn":  (-6.13,    5.0000),
}

# ---------- 2) Delta Vz Windows (ΔVz = Vz(e−) − Vz(π+)) ----------
# OLD — loose placeholder, needed update:
# DVZ_WINDOWS_OB = {
#     "LD2": (-20.000, 20.0000),
#     "CxC": (-20.000, 20.0000),
#     "Cu":  (-20.000, 20.0000),
#     "Sn":  (-20.000, 20.0000),
# }
# NEW — Table 7 (π+ OB, μ±3σ) from RGD_CommonNote2
DVZ_WINDOWS_OB = {
    "LD2": (-3.99, 2.29),
    "CxC": (-3.98, 2.30),
    "Cu":  (-3.95, 2.26),
    "Sn":  (-3.93, 2.30),
}

DVZ_WINDOWS_IB = {
    "LD2": (-20.000, 20.0000),
    "CxC": (-20.000, 20.0000),
    "Cu":  (-20.000, 20.0000),
    "Sn":  (-20.000, 20.0000),
}

# ---------- 3) DC Edge Cuts (π+) ----------
# NEW — Section 3.4 from RGD_CommonNote2
# π+: R1=3.875 cm, R2=4.375 cm, R3=2.625 cm
EDGE_CUTS_PIP_OB = {1: 3.875, 2: 4.375, 3: 2.625}

# ---------- 4) Helper Functions ----------

def vz_mask(vz, target, polarity="OB"):
    """Vertex Z cut based on target and polarity."""
    windows = VZ_WINDOWS_OB if polarity == "OB" else VZ_WINDOWS_IB
    vz_min, vz_max = windows[target]
    return (vz >= vz_min) & (vz <= vz_max)

def dvz_mask(pip_vz, ele_vz, target, polarity="OB"):
    """Delta Vz matching using target-specific dictionaries."""
    windows = DVZ_WINDOWS_OB if polarity == "OB" else DVZ_WINDOWS_IB
    dvz_min, dvz_max = windows[target]
    dvz = np.asarray(pip_vz) - ele_vz
    return (dvz >= dvz_min) & (dvz <= dvz_max)

def chi2pid_mask(chi2pid, cut_val=10.0):
    """Refined Pion identification."""
    return np.abs(np.asarray(chi2pid)) < cut_val

# Delta t cut coefficients for π+ in outbending polarity.
# These come from Mathieu Ouillon's implementation of the RG-D note cuts.
DELTAT_COEFFS_OB = {
    "LD2": {
        "mean":  [0.000922, -0.010435, 0.046222],
        "sigma": [0.002655, -0.021940, 0.114758],
    },
    "CxC": {
        "mean":  [-0.000063, -0.004623, 0.039045],
        "sigma": [0.001879, -0.017147, 0.108605],
    },
    "Cu": {
        "mean":  [0.001801, -0.015591, 0.051889],
        "sigma": [0.003742, -0.027802, 0.120968],
    },
    "Sn": {
        "mean":  [-0.000403, -0.001866, 0.031458],
        "sigma": [0.001510, -0.014369, 0.104379],
    },
}

def delt_t_mask(p, pip_dt, target, polarity="OB", n_sigma=3.0):
    """Δt cut using RG-D polynomial parameterization for π+."""
    if polarity != "OB":
        raise ValueError("Delta-t coefficients currently only implemented for OB polarity")

    coeffs = DELTAT_COEFFS_OB[target]
    p = np.asarray(p)
    pip_dt = np.asarray(pip_dt)

    mean = coeffs["mean"][0] * p**2 + coeffs["mean"][1] * p + coeffs["mean"][2]
    sigma = coeffs["sigma"][0] * p**2 + coeffs["sigma"][1] * p + coeffs["sigma"][2]

    cut_low = mean - n_sigma * sigma
    cut_high = mean + n_sigma * sigma

    valid = np.isfinite(pip_dt) & np.isfinite(p)
    result = np.zeros_like(pip_dt, dtype=bool)
    result[valid] = (pip_dt[valid] >= cut_low[valid]) & (pip_dt[valid] <= cut_high[valid])
    return result

def dc_edge_mask_pip(df, polarity="OB"):
    """DC fiducial cut for π+ tracks (Section 3.4, RGD_CommonNote2)."""
    cuts = EDGE_CUTS_PIP_OB  # only OB for now
    mask_r1 = (df["dc_edge_r1"] > cuts[1]) | (df["dc_edge_r1"] == 0)
    mask_r2 = (df["dc_edge_r2"] > cuts[2]) | (df["dc_edge_r2"] == 0)
    mask_r3 = (df["dc_edge_r3"] > cuts[3]) | (df["dc_edge_r3"] == 0)
    return mask_r1 & mask_r2 & mask_r3

# ---------- 5) The Master Cutflow Function ----------

def pip_cutflow(df, ele_vz, target, polarity="OB"):
    """
    Master function to identify pions and calculate cutflow stats.
    """
    pid     = df["pid"].to_numpy()
    status  = df["status"].to_numpy()
    # Keep chi2pid as a diagnostic/history field only.
    chi2pid = df["chi2pid"].to_numpy()
    pip_dt  = df["pip_dt"].to_numpy()
    p       = df["p"].to_numpy()
    vz      = df["vz"].to_numpy()
    
    masks = {}

    # STEP 1: BASE (PID 211 + FD Status)
    masks["base"] = is_fd_pip(pid, status)
    N_base = int(np.sum(masks["base"]))

    # STEP 2: chi2pid cut
    masks["chi2pid"] = masks["base"] & chi2pid_mask(chi2pid, cut_val=10.0)

    # STEP 3: Δt vs p cut
    masks["deltat"] = masks["chi2pid"] & delt_t_mask(p, pip_dt, target, polarity)

    # STEP 4: Vz
    masks["vz"] = masks["deltat"] & vz_mask(vz, target, polarity)

    # STEP 5: DC edge cut
    masks["dc"] = masks["vz"] & dc_edge_mask_pip(df, polarity)

    # STEP 6: ΔVz
    masks["dvz"] = masks["dc"] & dvz_mask(vz, ele_vz, target, polarity)


    masks["final"] = masks["dvz"]

    # Generate Stats (Consistent dictionary keys)
    order = ["base", "chi2pid", "deltat", "vz", "dc", "dvz", "final"]
    cutflow = {step: {"N": int(np.sum(masks[step])), 
                      "eff_base": 100.0 * np.sum(masks[step]) / N_base if N_base > 0 else 0.0} 
               for step in order}

    return masks["final"], cutflow, masks