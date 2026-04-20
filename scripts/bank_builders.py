"""
bank_builders.py (Vectorized)

High-performance builder for CLAS12 DataFrames.
Includes internal Sampling Fraction calculation to avoid redundant array flattening.
"""

import numpy as np
import pandas as pd
import awkward as ak

from physics import get_p, get_phi, get_sector
from physics_constants import M_PION_PLUS, C_VEL

# ---------------------------
# 1) Branch list
# ---------------------------
REC_BRANCHES = [
    # Event info
    "RUN_config_run",
    "RUN_config_event",
    "REC_Event_helicity",
    "REC_Event_helicityRaw",
    # Particle info
    "REC_Particle_pid",
    "REC_Particle_charge",
    "REC_Particle_px",
    "REC_Particle_py",
    "REC_Particle_pz",
    "REC_Particle_status",
    "REC_Particle_vz",
    "REC_Particle_vx",
    "REC_Particle_vy",
    "REC_Particle_beta",
    "REC_Particle_chi2pid",
    "REC_Particle_vt",  # RF and z corrected vertex time
    # Scintillator Bank (FTOF)
    "REC_Scintillator_pindex",
    "REC_Scintillator_detector",
    "REC_Scintillator_layer",
    "REC_Scintillator_time",
    "REC_Scintillator_path",
    "REC_Scintillator_sector",
    "REC_Scintillator_component",
    # Cherenkov
    "REC_Cherenkov_pindex",
    "REC_Cherenkov_nphe",
    "REC_Cherenkov_detector",
    # Calorimeter
    "REC_Calorimeter_pindex",
    "REC_Calorimeter_detector",
    "REC_Calorimeter_layer",
    "REC_Calorimeter_energy",
    "REC_Calorimeter_lv",
    "REC_Calorimeter_lw",
    # DC Trajectory
    "REC_Traj_detector",
    "REC_Traj_layer",
    "REC_Traj_pindex",
    "REC_Traj_edge",
    "REC_Traj_x",
    "REC_Traj_y",
]

MC_BRANCHES = [
    "MC_Particle_pid",
    "MC_Particle_px", 
    "MC_Particle_py", 
    "MC_Particle_pz", 
    "MC_Particle_vx", 
    "MC_Particle_vy", 
    "MC_Particle_vz", 
    "MC_Particle_vt",
    "MC_RecMatch_pindex",
    "MC_RecMatch_mcindex",
    "MC_RecMatch_quality",
]

# ---------------------------
# 2) Generator-Level Builder
# ---------------------------

def build_gen_arrays(arrs):
    """
    Build a generator-level DataFrame from MC_Particle bank.

    One row per (scattered e, pi+) pair per event — mirrors the reco
    parquet structure where all pions passing cuts are kept.
    The scattered electron (first pid=11) is replicated for each pi+
    in the same event, so multi-pion events produce multiple rows.
    """
    mc_pid = arrs["MC_Particle_pid"]
    mc_px  = arrs["MC_Particle_px"]
    mc_py  = arrs["MC_Particle_py"]
    mc_pz  = arrs["MC_Particle_pz"]
    mc_vz  = arrs["MC_Particle_vz"]

    e_mask   = (mc_pid == 11)
    pip_mask = (mc_pid == 211)

    # Scattered electron: take first pid=11 per event (option[float32] per event)
    e_px_j = ak.firsts(mc_px[e_mask])
    e_py_j = ak.firsts(mc_py[e_mask])
    e_pz_j = ak.firsts(mc_pz[e_mask])
    e_vz_j = ak.firsts(mc_vz[e_mask])

    # All pi+ per event (var * float32 per event)
    pip_px_j = mc_px[pip_mask]
    pip_py_j = mc_py[pip_mask]
    pip_pz_j = mc_pz[pip_mask]

    # Broadcast the per-event electron scalar to match each pion in the event.
    # Events with no pion contribute 0 rows; events with N pions contribute N rows.
    e_px_b,  pip_px_b = ak.broadcast_arrays(e_px_j,  pip_px_j)
    e_py_b,  pip_py_b = ak.broadcast_arrays(e_py_j,  pip_py_j)
    e_pz_b,  pip_pz_b = ak.broadcast_arrays(e_pz_j,  pip_pz_j)
    e_vz_b,  _        = ak.broadcast_arrays(e_vz_j,  pip_px_j)

    # Event index for every row (needed to align gen/reco later)
    event_idx = ak.to_numpy(
        ak.flatten(ak.broadcast_arrays(ak.local_index(pip_px_j, axis=0), pip_px_j)[0])
    )

    def flat_np(arr):
        return ak.to_numpy(ak.fill_none(ak.flatten(arr), np.nan)).astype(np.float32)

    e_px  = flat_np(e_px_b);   e_py  = flat_np(e_py_b);   e_pz  = flat_np(e_pz_b)
    e_vz  = flat_np(e_vz_b)
    pip_px = flat_np(pip_px_b); pip_py = flat_np(pip_py_b); pip_pz = flat_np(pip_pz_b)

    # Drop rows where the event had no scattered electron
    valid = np.isfinite(e_px)

    return pd.DataFrame({
        "event_idx": event_idx[valid],
        "e_px":   e_px[valid],   "e_py":   e_py[valid],   "e_pz":   e_pz[valid],
        "e_vz":   e_vz[valid],
        "pip_px": pip_px[valid], "pip_py": pip_py[valid], "pip_pz": pip_pz[valid],
    })

# ---------------------------
# 3) Helper: Map Hits to Particles
# ---------------------------
def map_hits_to_particles_vectorized(arrs, bank_prefix, hit_mask, value_branch, total_particles, particle_offsets):
    """
    Vectorized map of satellite bank values to main Particle bank.
    Uses scatter-add to ensure hits within the same mask are summed, not erased.
    """
    pindex_jagged = arrs[f"{bank_prefix}_pindex"]
    value_jagged  = arrs[f"{bank_prefix}_{value_branch}"]
    
    # 1. Get Event Index for every hit
    event_indices = ak.flatten(
        ak.broadcast_arrays(ak.local_index(pindex_jagged, axis=0), pindex_jagged)[0][hit_mask]
    ).to_numpy()
    
    # 2. Flatten data
    flat_pindex = ak.flatten(pindex_jagged[hit_mask]).to_numpy()
    flat_value  = ak.flatten(value_jagged[hit_mask]).to_numpy()
    
    # 3. Global Index calculation
    global_indices = particle_offsets[event_indices] + flat_pindex
    
    # 4. Create empty array
    result = np.zeros(total_particles, dtype=np.float32)
    valid_idx = (global_indices >= 0) & (global_indices < total_particles)
    
    # --- THE MODIFICATION ---
    # We use np.add.at instead of = so we don't erase data if a particle 
    # hits two sensors in the same detector layer.
    np.add.at(result, global_indices[valid_idx], flat_value[valid_idx])
    
    return result
# ---------------------------
# 4) Main Builder
# ---------------------------

def build_per_particle_arrays(arrs, target_group="LD2"):
    """
    Vectorized construction of the flat DataFrame for CLAS12 analysis.
    """
    # --- A. Setup Backbone (REC::Particle) ---
    pid_jagged = arrs["REC_Particle_pid"]
    counts = ak.num(pid_jagged)
    total_particles = np.sum(counts)
    
    # Offset array: Index where event i starts in the flat array
    offsets = np.concatenate(([0], np.cumsum(counts.to_numpy())[:-1]))
    
    # Flatten Basic Kinematics
    pid    = ak.flatten(pid_jagged).to_numpy()
    charge = ak.flatten(arrs["REC_Particle_charge"]).to_numpy()
    px     = ak.flatten(arrs["REC_Particle_px"]).to_numpy()
    py     = ak.flatten(arrs["REC_Particle_py"]).to_numpy()
    pz     = ak.flatten(arrs["REC_Particle_pz"]).to_numpy()
    status = ak.flatten(arrs["REC_Particle_status"]).to_numpy()
    vz     = ak.flatten(arrs["REC_Particle_vz"]).to_numpy()
    beta    = ak.flatten(arrs["REC_Particle_beta"]).to_numpy()  
    chi2pid = ak.flatten(arrs["REC_Particle_chi2pid"]).to_numpy()
    
    run_num = np.repeat(arrs["RUN_config_run"].to_numpy(), counts.to_numpy())

    # --- B. Calculate Kinematics (Needed for SF and Sector) ---
    p_flat = get_p(px, py, pz)
    phi_flat = get_phi(px, py, degrees=True)
    sec_flat = get_sector(phi_flat)

    # --- C. Map HTCC (Nphe) ---
    # Sums multiple hits if an electron hits two mirrors (mirror crack)
    cher_det = arrs["REC_Cherenkov_detector"]
    # HTCC (Detector 15)
    mask_htcc = (cher_det == 15)
    nphe_htcc = map_hits_to_particles_vectorized(arrs, "REC_Cherenkov", mask_htcc, "nphe", total_particles, offsets)

    # LTCC (Detector 16)
    mask_ltcc = (cher_det == 16)
    nphe_ltcc = map_hits_to_particles_vectorized(arrs, "REC_Cherenkov", mask_ltcc, "nphe", total_particles, offsets)

    # --- D. Map Calorimeter Layers (Separately) ---
    calo_det = arrs["REC_Calorimeter_detector"]
    calo_lay = arrs["REC_Calorimeter_layer"]
    
    # Define Masks for PCAL (1), ECIN (4), ECOUT (7)
    # Note: Det 7 is ECAL
    m_pcal  = (calo_det == 7) & (calo_lay == 1)
    m_ecin  = (calo_det == 7) & (calo_lay == 4)
    m_ecout = (calo_det == 7) & (calo_lay == 7)
    
    # Get Energy per layer (Keeps them separate as requested)
    e_pcal  = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", m_pcal,  "energy", total_particles, offsets)
    e_ecin  = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", m_ecin,  "energy", total_particles, offsets)
    e_ecout = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", m_ecout, "energy", total_particles, offsets)
    
    # Get PCAL coordinate for fiducial cuts
    v_pcal = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", m_pcal, "lv", total_particles, offsets)
    w_pcal = map_hits_to_particles_vectorized(arrs, "REC_Calorimeter", m_pcal, "lw", total_particles, offsets)

    # --- E. Sampling Fraction (SF) Calculation ---
    # SF = (Total Energy in Calorimeter) / Momentum
    # Since we use map_hits_to_particles_vectorized, we can just sum the flat arrays
    Etot_flat = e_pcal + e_ecin + e_ecout
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sf_flat = np.where(p_flat > 0, Etot_flat / p_flat, np.nan)

    # --- F. Map DC Trajectory (Det 6) ---
    traj_det = arrs["REC_Traj_detector"]
    traj_lay = arrs["REC_Traj_layer"]
    mask_dc = (traj_det == 6)

    def map_dc(m, branch):
        return map_hits_to_particles_vectorized(arrs, "REC_Traj", m, branch, total_particles, offsets)
    
    # 1. Create separate masks for 6, 18, 36 (Verified layers)
    mask_r1 = mask_dc & (traj_lay == 6)
    mask_r2 = mask_dc & (traj_lay == 18)
    mask_r3 = mask_dc & (traj_lay == 36)
    
    # 2. Map coordinates and edges for each region specifically
    x_r1, y_r1, edge_r1 = map_dc(mask_r1, "x"), map_dc(mask_r1, "y"), map_dc(mask_r1, "edge")
    x_r2, y_r2, edge_r2 = map_dc(mask_r2, "x"), map_dc(mask_r2, "y"), map_dc(mask_r2, "edge")
    x_r3, y_r3, edge_r3 = map_dc(mask_r3, "x"), map_dc(mask_r3, "y"), map_dc(mask_r3, "edge")

    # --- G. Timing Mapping (FTOF Priority: 2 -> 1 -> 3) ---
    # We must ensure map_hits_to_particles returns 0.0 for "No Hit" 
    # but keeps the data for "Out of Time" hits.

    # 1. Get raw banks
    scin_det = arrs["REC_Scintillator_detector"]
    scin_lay = arrs["REC_Scintillator_layer"]

    # 2. Priority layers separately
    # map_hits_to_particles_vectorized must return the RAW time from the bank
    t2 = map_hits_to_particles_vectorized(arrs, "REC_Scintillator", (scin_det == 12) & (scin_lay == 2), "time", total_particles, offsets)
    p2 = map_hits_to_particles_vectorized(arrs, "REC_Scintillator", (scin_det == 12) & (scin_lay == 2), "path", total_particles, offsets)

    t1 = map_hits_to_particles_vectorized(arrs, "REC_Scintillator", (scin_det == 12) & (scin_lay == 1), "time", total_particles, offsets)
    p1 = map_hits_to_particles_vectorized(arrs, "REC_Scintillator", (scin_det == 12) & (scin_lay == 1), "path", total_particles, offsets)

    t3 = map_hits_to_particles_vectorized(arrs, "REC_Scintillator", (scin_det == 12) & (scin_lay == 3), "time", total_particles, offsets)
    p3 = map_hits_to_particles_vectorized(arrs, "REC_Scintillator", (scin_det == 12) & (scin_lay == 3), "path", total_particles, offsets)

    # 3. PRIORITY LOGIC (C++ Replica)
    # We choose T and P based on the existence of Layer 2, then 1, then 3.
    best_t = np.zeros(total_particles)
    best_p = np.zeros(total_particles)

    # ADD THIS LINE TO FIX THE NAMEERROR:
    # This creates an array that stores the layer number actually used for each track
    used_lay = np.where(t2 > 0, 2, np.where(t1 > 0, 1, np.where(t3 > 0, 3, 0)))

    # If Layer 3 has data, take it.
    mask3 = (t3 > 0)
    best_t[mask3] = t3[mask3]
    best_p[mask3] = p3[mask3]

    # If Layer 1 has data, it overwrites Layer 3.
    mask1 = (t1 > 0)
    best_t[mask1] = t1[mask1]
    best_p[mask1] = p1[mask1]

    # If Layer 2 has data, it overwrites everything else.
    mask2 = (t2 > 0)
    best_t[mask2] = t2[mask2]
    best_p[mask2] = p2[mask2]

    # --- H. Delta T Calculation (No aggressive cutting) ---
    vt_particle = ak.flatten(arrs["REC_Particle_vt"]).to_numpy()
    beta_theory = p_flat / np.sqrt(p_flat**2 + M_PION_PLUS**2)

    # Calculation mask - ONLY filter out cases where there is NO detector hit at all
    # and ensure p_flat > 0 to avoid division by zero.
    has_hit = (best_t > 0) & (p_flat > 0)

    pip_dt = np.full(total_particles, 99999.0, dtype=np.float32)

    # Use r"..." to avoid syntax warnings in your labels later
    # Delta_t = (time - path/(c*beta)) - vt
    pip_dt[has_hit] = (best_t[has_hit] - (best_p[has_hit] / (C_VEL * beta_theory[has_hit]))) - vt_particle[has_hit]

    # ---G. Helicity Information for Azimuthal Modulation ---
    helicity = np.repeat(arrs["REC_Event_helicity"].to_numpy(), counts.to_numpy())
    helicityRaw = np.repeat(arrs["REC_Event_helicityRaw"].to_numpy(), counts.to_numpy())

    # --- H. Build Final DataFrame ---
    df = pd.DataFrame({
        # Event Info
        "event_id": np.repeat(arrs["RUN_config_event"].to_numpy(), counts.to_numpy()),
        "run":      run_num,
        "event_idx_local": np.repeat(np.arange(len(counts)), counts),
        "helicity": helicity, 
        "helicityRaw": helicityRaw,  
        # Rec Particle
        "pid":    pid,
        "charge": charge,
        "px":     px, "py": py, "pz": pz,
        "status": status,
        "vz":     vz,
        "p":      p_flat,
        "phi":    phi_flat,
        "sector": sec_flat,
        "sf":     sf_flat,
        "beta":   beta,   
        "chi2pid": chi2pid,
        "vt":       vt_particle, 
        "pip_dt":   pip_dt,      
        "ftof_lay": used_lay,   

        # Cherenkov
        "Nphe_htcc": nphe_htcc,
        "Nphe_ltcc": nphe_ltcc,
        
        # Calorimeter
        "E_pcal":  e_pcal,
        "E_ecin":  e_ecin,
        "E_ecout": e_ecout,
        "v_pcal":  v_pcal, 
        "w_pcal":  w_pcal,
        
        # DC Trajectory (Matching names from Section F)
        "dc_x_r1": x_r1, "dc_y_r1": y_r1, "dc_edge_r1": edge_r1,
        "dc_x_r2": x_r2, "dc_y_r2": y_r2, "dc_edge_r2": edge_r2,
        "dc_x_r3": x_r3, "dc_y_r3": y_r3, "dc_edge_r3": edge_r3,
    })

    return df