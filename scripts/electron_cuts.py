import numpy as np
from params_electron_sf_outbending import SF_PARAMS_OB 
from common_cuts import is_fd_ele
# from params_electron_sf_inbending import SF_PARAMS_IB # <--- Uncomment when you have this file

# ---------- 1) Vz Windows (RG-D Outbending) ----------
VZ_WINDOWS_OB = {
    "LD2": (-15.0,  5.0),
    "CxC": (-10.6,  5.0),
    "Cu":  (-10.6, -6.5),
    "Sn":  (-5.5,   5.0),
    "CuSn":(-10.6,  5.0),
}

# ---------- 2) DC Edge Cuts ----------
# OLD values (per-target, slightly off from note averages):
# EDGE_CUTS_OB = {
#     "LD2":  {1: 1.68, 2: 2.00, 3: 8.75},
#     "CxC":  {1: 1.70, 2: 2.02, 3: 8.92},
#     "CuSn": {1: 1.69, 2: 2.00, 3: 8.89},
# }
# NEW — Table 3 averages from RGD_CommonNote2: R1=1.88, R2=2.08, R3=8.62 (same for all targets)
EDGE_CUTS_OB = {
    "LD2":  {1: 1.88, 2: 2.08, 3: 8.62},
    "CxC":  {1: 1.88, 2: 2.08, 3: 8.62},
    "CuSn": {1: 1.88, 2: 2.08, 3: 8.62},
}

# ---------- 3) Helper Functions ----------
def sf_mean(p, coeffs):
    """Evaluates polynomial: a + b*p + c*p^2 + d*p^3"""
    a, b, c, d = coeffs
    return a + b*p + c*p*p + d*p*p*p

def sf_sigma(p, coeffs):
    """Evaluates polynomial: a + b*p + c*p^2 + d*p^3"""
    a, b, c, d = coeffs
    return a + b*p + c*p*p + d*p*p*p

def sf_cut_mask(p, sf, sector, target, polarity="OB"):
    if polarity == "OB":
        params_dict = SF_PARAMS_OB
    else:
        raise NotImplementedError(f"Polarity {polarity} SF parameters are not yet loaded!")

    # --- TARGET MAPPING ---
    # Use LD2 parameters for Cu because the CuSn set droops 
    # while the calibrated Copper data is flat.
    # --- TARGET MAPPING ---
    # This is the new "Weapon" logic
    if target in ["Cu", "Sn", "CuSn"]:
        target_key = "my_CuSn"
    
    # We commented out the old way below using '#' so it doesn't crash
    # if target == "Cu":
    #     target_key = "CuSn" 
    # elif target == "Sn":
    #     target_key = "CuSn"
    
    else:
        target_key = target

    p = np.asarray(p)
    sf = np.asarray(sf)
    sector = np.asarray(sector)
    mask = np.zeros(len(p), dtype=bool)

    for s in range(1, 7):
        idx = (sector == s)
        if not np.any(idx): continue
        
        if target_key not in params_dict:
            pars = params_dict["LD2"][s]
        else:
            pars = params_dict[target_key][s]

        mu  = sf_mean(p[idx], pars["mu"])
        sig = sf_sigma(p[idx], pars["sigma"])
        
        # OLD: 3-Sigma Cut
        # mask[idx] = (sf[idx] >= mu - 3*sig) & (sf[idx] <= mu + 3*sig)
        # NEW — RGD_CommonNote2 Table 4 specifies ±3.5σ
        mask[idx] = (sf[idx] >= mu - 3.5*sig) & (sf[idx] <= mu + 3.5*sig)
        
    return mask

def vz_mask(vz, target):
    vz_min, vz_max = VZ_WINDOWS_OB[target]
    return (vz >= vz_min) & (vz <= vz_max)

def dc_edge_mask(df, target_group):
    # Get the cut values (e.g., 1.69cm for R1, 8.89cm for R3)
    cuts = EDGE_CUTS_OB[target_group]
    
    # A track is good only if it passes the edge cut in ALL regions it hit
    mask_r1 = (df["dc_edge_r1"] > cuts[1]) | (df["dc_edge_r1"] == 0)
    mask_r2 = (df["dc_edge_r2"] > cuts[2]) | (df["dc_edge_r2"] == 0)
    mask_r3 = (df["dc_edge_r3"] > cuts[3]) | (df["dc_edge_r3"] == 0)
    
    return mask_r1 & mask_r2 & mask_r3

def calo_fid_mask(v, w):
    return (v > 9.0) & (w > 9.0)

def pcal_energy_cut(epcal, e_min=0.06):
    return epcal > e_min

# ---------- 4) The Master Cutflow Function ----------

def electron_cutflow(df, target, polarity="OB", sample_type="data"):
    # Extract arrays
    pid    = df["pid"].to_numpy()
    status = df["status"].to_numpy()
    vz     = df["vz"].to_numpy()
    Nphe   = df["Nphe_htcc"].to_numpy() 
    p      = df["p"].to_numpy()
    sf     = df["sf"].to_numpy()
    sector = df["sector"].to_numpy()
    v_pcal = df["v_pcal"].to_numpy()
    w_pcal = df["w_pcal"].to_numpy()
    E_pcal = df["E_pcal"].to_numpy()   


    target_group = "CuSn" if target in ["Cu", "Sn", "CuSn"] else target
    masks = {}

    # Initial Candidate Selection (PID 11 + FD + '-ve' Status)
    candidate_mask = is_fd_ele(pid, status)

    # --- STEP 1: BASE (PID + Status + NPHE) ---
    masks["base"] = candidate_mask & (Nphe > 2)
    
     # Use N_base as the reference for percentages in the table
    N_base = int(np.sum(masks["base"]))

    # --- STEP 2: MOMENTUM CUT (P > 1.0 GeV) ---
    masks["p"] = masks["base"] & (p > 0.8)
    
   

    # --- STEP 3: SAMPLING FRACTION (Before Vz) ---
    # This allows checking the detector response for the whole target region
    sf_ok = sf_cut_mask(p, sf, sector, target, polarity)
    masks["sf"] = masks["p"] & sf_ok
    
    # --- STEP 4: DETECTOR CUTS (PCAL & DC) ---
    masks["pcal"] = masks["sf"] & pcal_energy_cut(E_pcal) & calo_fid_mask(v_pcal, w_pcal)
    masks["dc"]   = masks["pcal"] & dc_edge_mask(df, target_group)
    
    # --- STEP 5: VERTEX CUT (Final Foil Isolation) ---
    masks["vz"]    = masks["dc"] & vz_mask(vz, target)
    masks["final"] = masks["vz"]

    # Define the order for the terminal printout
    order = ["base", "p", "sf", "pcal", "dc", "vz", "final"]
    
    cutflow = {}
    for step in order:
        N = int(np.sum(masks[step]))
        #Efficiency is now relative to the raw 'Base' (PID 11 + NPHE)
        eff = 100.0 * N / N_base if N_base > 0 else 0.0
        cutflow[step] = {"N": N, "eff_base": eff}

    return masks["final"], cutflow, masks