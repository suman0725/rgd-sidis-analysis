# truth_matching.py

"""
Truth matching helpers for CLAS12 MC samples.

WORKFLOW:
  1. add_truth_matching(df, arrs)              → mc_pid, match_quality, mcindex
  2. enforce_truth_pid_matching(df, q_min)    → mc_pid_raw, match_quality_raw,
                                                  mcindex_raw, expected_mc_pid,
                                                  truth_pid_ok, truth_matched
  3. add_angular_matching_diagnostic(df, arrs) → ang_match_pid, ang_match_quality,
                                                  ang_match_mcindex, delta_theta/phi,
                                                  ang_match_pass, ang_pid_ok,
                                                  ang_truth_like, match_agreement
  4. Filter to df[truth_matched == 1]

PRIMARY MATCHING (COATJAVA official):
  Uses MC::RecMatch bank: pindex → mcindex, quality, mc_pid.
  COATJAVA remains official truth; other methods are diagnostics only.

DIAGNOSTIC MATCHING (RichCap-style angular):
  Optional: matches by kinematic proximity (Δθ, Δφ, Δp/p).
  Useful for cross-validation and understanding systematic differences.
"""

import numpy as np
import awkward as ak
import math


def add_truth_matching(df_all, arrs, quality_min=None):
    """
    Attach truth-matching info to df_all for MC samples.

    Parameters
    ----------
    df_all : pandas.DataFrame
        Flat per-REC-track DataFrame from build_per_particle_arrays().
        Rows are in the same order as flattening REC_Particle_* in arrs.
    arrs : awkward.highlevel.Array or dict-like
        uproot arrays dict, must contain (for sim files):
          - MC_RecMatch_pindex
          - MC_RecMatch_mcindex
          - MC_RecMatch_quality
          - MC_Particle_pid
    quality_min : float or None
        If not None, keep only rows with match_quality >= quality_min.

    Returns
    -------
    df_out : pandas.DataFrame
        df_all with extra columns:
           - mc_pid: matched MC particle PID
           - match_quality: COATJAVA match quality score
           - mcindex: MC::Particle index of the match
    """

    # Event-wise arrays
    rec_pid    = arrs["REC_Particle_pid"]          # just for event sizes
    rm_pindex  = arrs["MC_RecMatch_pindex"]
    rm_mcindex = arrs["MC_RecMatch_mcindex"]
    rm_quality = arrs["MC_RecMatch_quality"]
    mc_pid_arr = arrs["MC_Particle_pid"]

    mc_pid_per_event        = []
    match_quality_per_event = []
    mcindex_per_event       = []

    # Loop over events (one "row" per event in each awkward array)
    for pid_ev, pidx_ev, mcidx_ev, qual_ev, mc_pid_ev in zip(
        rec_pid, rm_pindex, rm_mcindex, rm_quality, mc_pid_arr
    ):
        n_part = len(pid_ev)

        # Default: no match
        mc_pid_evt   = np.zeros(n_part, dtype=np.int32)
        qual_evt_out = np.zeros(n_part, dtype=np.float32)
        mcidx_evt    = np.full(n_part, -1, dtype=np.int32)  # -1 means unmatched (0 is valid index)

        # Loop over rows of MC::RecMatch for this event
        # pidx_ev[i]: REC::Particle index (iRec)
        # mcidx_ev[i]: MC::Particle index
        # qual_ev[i]: match quality
        for pidx, mcind, q in zip(pidx_ev, mcidx_ev, qual_ev):
            if pidx < 0 or pidx >= n_part:
                continue

            # attach quality and mcindex only if mcind is valid
            if 0 <= mcind < len(mc_pid_ev):
                qual_evt_out[pidx] = q
                mcidx_evt[pidx] = mcind
                mc_pid_evt[pidx] = mc_pid_ev[mcind]
            else:
                mc_pid_evt[pidx] = 0  # no valid match

        mc_pid_per_event.append(mc_pid_evt)
        match_quality_per_event.append(qual_evt_out)
        mcindex_per_event.append(mcidx_evt)

    # Flatten to match flat REC_Particle ordering used in df_all
    mc_pid_flat        = ak.to_numpy(ak.flatten(ak.Array(mc_pid_per_event)))
    match_quality_flat = ak.to_numpy(ak.flatten(ak.Array(match_quality_per_event)))
    mcindex_flat       = ak.to_numpy(ak.flatten(ak.Array(mcindex_per_event)))

    if mc_pid_flat.shape[0] != len(df_all):
        raise RuntimeError(
            f"Truth-matching length mismatch: mc_pid_flat={mc_pid_flat.shape[0]}, df_all={len(df_all)}"
        )

    df_all = df_all.copy()
    df_all["mc_pid"] = mc_pid_flat
    df_all["match_quality"] = match_quality_flat
    df_all["mcindex"] = mcindex_flat

    # Note: quality_min filtering is now handled by enforce_truth_pid_matching() or user code
    # to keep raw COATJAVA data and allow custom filtering logic

    return df_all


def enforce_truth_pid_matching(df_all, quality_min=None):
    """
    Validate COATJAVA truth matches against expected MC PID by REC particle type.

    Enforces that:
      - REC electron (PID 11) must match MC PID 11
      - REC π+ (PID 211) must match MC PID 211

    Works on flat pandas DataFrame directly.

    Parameters
    ----------
    df_all : pandas.DataFrame
        Output from add_truth_matching(), must have columns:
          - REC_Particle_pid (or 'pid'): REC particle PID for determining expectation
          - mc_pid: matched MC PID from COATJAVA
          - match_quality: match quality from COATJAVA
    quality_min : float or None
        Minimum match quality threshold for truth_matched flag.

    Returns
    -------
    df_validated : pandas.DataFrame
        df_all with new columns:
          - mc_pid_raw: original COATJAVA match PID (copy of mc_pid)
          - match_quality_raw: original COATJAVA match quality (copy of match_quality)
          - mcindex_raw: original MC::Particle index (copy of mcindex)
          - expected_mc_pid: PID expected by REC particle type (11 or 211)
          - truth_pid_ok: 1 if mc_pid_raw == expected_mc_pid, 0 otherwise
          - truth_matched: 1 if truth_pid_ok AND quality >= quality_min
    """

    df_out = df_all.copy()

    # Rename original columns to "raw"
    if "mc_pid" in df_out.columns:
        df_out["mc_pid_raw"] = df_out["mc_pid"].copy()
    if "match_quality" in df_out.columns:
        df_out["match_quality_raw"] = df_out["match_quality"].copy()
    if "mcindex" in df_out.columns:
        df_out["mcindex_raw"] = df_out["mcindex"].copy()

    # Get REC particle PID column (try both naming conventions)
    if "REC_Particle_pid" in df_out.columns:
        rec_pid_col = df_out["REC_Particle_pid"].values
    elif "pid" in df_out.columns:
        rec_pid_col = df_out["pid"].values
    else:
        raise ValueError("Column 'REC_Particle_pid' or 'pid' not found in df_all")

    # Determine expected MC PID for each row
    expected_mc_pid = np.zeros(len(df_out), dtype=np.int32)
    for i, pid_rec in enumerate(rec_pid_col):
        if pid_rec == 11:  # electron
            expected_mc_pid[i] = 11
        elif pid_rec == 211:  # pi+
            expected_mc_pid[i] = 211
        # else: expected_mc_pid[i] = 0 (already initialized)

    df_out["expected_mc_pid"] = expected_mc_pid

    # Check if matched PID agrees with expectation
    if "mc_pid_raw" in df_out.columns:
        df_out["truth_pid_ok"] = (df_out["mc_pid_raw"] == df_out["expected_mc_pid"]).astype(int)
    else:
        df_out["truth_pid_ok"] = 0

    # Final truth_matched flag: PID correct AND quality passes threshold
    if quality_min is not None and "match_quality_raw" in df_out.columns:
        df_out["truth_matched"] = (
            (df_out["truth_pid_ok"] == 1) &
            (df_out["match_quality_raw"] >= quality_min)
        ).astype(int)
    else:
        df_out["truth_matched"] = df_out["truth_pid_ok"].copy()

    return df_out


def ask_truth_match_options(sample_type):
    """
    For simulation samples, ask whether to apply a REC↔MC quality cut
    and what minimum quality to use.

    Returns
    -------
    (apply_tm, q_min)
      apply_tm : bool (True if we should do truth matching)
      q_min    : float or None  (minimum quality if apply_tm is True)
    """
    if sample_type != "sim":
        return False, None

    ans = input("\nApply MC truth-matching cut on simulation? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        return False, None

    q_str = input("  Minimum match quality (e.g. 0.98): ").strip()
    try:
        q_min = float(q_str) if q_str else 0.98
    except ValueError:
        print("  Could not parse number, defaulting to 0.98")
        q_min = 0.98

    print(f"  -> will apply truth-matching with quality >= {q_min}")
    return True, q_min


def add_angular_matching_diagnostic(df_all, arrs):
    """
    Add RichCap-style angular matching as diagnostic columns.

    Matches REC particles to MC particles by minimizing angular distance
    (Δθ, Δφ) and momentum difference, compared to expected MC PID by particle type.

    Parameters
    ----------
    df_all : pandas.DataFrame
        Output from enforce_truth_pid_matching(), must have:
          - REC_Particle_px, REC_Particle_py, REC_Particle_pz
          - REC_Particle_pid (or 'pid'): for determining expected MC PID
          - expected_mc_pid (optional): if present, matches only to this PID
    arrs : dict-like
        uproot arrays dict with:
          - MC_Particle_pid, MC_Particle_px, MC_Particle_py, MC_Particle_pz

    Returns
    -------
    df_with_diag : pandas.DataFrame
        df_all with new columns:
          - ang_match_pid: matched MC PID by angular method
          - ang_match_mcindex: matched MC::Particle index by angular method
          - ang_match_quality: angular quality score (lower=better)
          - delta_theta: Δθ in degrees
          - delta_phi: Δφ in degrees
          - delta_p_rel: relative momentum difference
          - ang_match_pass: 1 if Δθ < 6° AND Δφ < 10°, 0 otherwise
          - ang_pid_ok: 1 if ang_match_pid == expected MC PID, 0 otherwise
          - ang_truth_like: 1 if ang_match_pass AND ang_pid_ok, 0 otherwise
          - match_agreement: 1 if mcindex_raw == ang_match_mcindex (mcindex agreement)
    """

    # SAFETY CHECK: ensure df hasn't been filtered
    # df must have same number of rows as flattened REC_Particle_pid from arrs
    rec_pid_awk = arrs["REC_Particle_pid"]
    expected_flat_len = sum(len(x) for x in rec_pid_awk)
    if len(df_all) != expected_flat_len:
        raise ValueError(
            f"DataFrame length mismatch: df_all has {len(df_all)} rows but "
            f"flattened REC_Particle_pid has {expected_flat_len} rows. "
            f"Angular diagnostics must run BEFORE filtering. "
            f"Run this function after add_truth_matching() and enforce_truth_pid_matching() "
            f"but before filtering on truth_matched."
        )

    # Operate on flat DataFrame directly using stored 4-momentum
    df_out = df_all.copy()

    # Get expected MC PID per particle (REC type determines what we match to)
    if "expected_mc_pid" not in df_out.columns:
        # Fallback: determine from REC_Particle_pid or 'pid'
        if "REC_Particle_pid" in df_out.columns:
            rec_pid_col = df_out["REC_Particle_pid"].values
        elif "pid" in df_out.columns:
            rec_pid_col = df_out["pid"].values
        else:
            raise ValueError("Need either 'expected_mc_pid' column or 'REC_Particle_pid'/'pid'")

        expected_mc_pid_arr = np.zeros(len(df_out), dtype=np.int32)
        for i, pid_rec in enumerate(rec_pid_col):
            if pid_rec == 11:
                expected_mc_pid_arr[i] = 11
            elif pid_rec == 211:
                expected_mc_pid_arr[i] = 211
    else:
        expected_mc_pid_arr = df_out["expected_mc_pid"].values

   # Get REC 4-momentum
    if all(col in df_out.columns for col in ["REC_Particle_px", "REC_Particle_py", "REC_Particle_pz"]):
        rec_px = df_out["REC_Particle_px"].values
        rec_py = df_out["REC_Particle_py"].values
        rec_pz = df_out["REC_Particle_pz"].values
    elif all(col in df_out.columns for col in ["px", "py", "pz"]):
        rec_px = df_out["px"].values
        rec_py = df_out["py"].values
        rec_pz = df_out["pz"].values
    else:
        raise ValueError("Missing REC momentum columns: need REC_Particle_px/py/pz or px/py/pz")

    # Get MC 4-momentum from arrays (awkward, event-wise)
    mc_pid_awk  = arrs["MC_Particle_pid"]
    mc_px_awk   = arrs["MC_Particle_px"]
    mc_py_awk   = arrs["MC_Particle_py"]
    mc_pz_awk   = arrs["MC_Particle_pz"]

    # Initialize output columns (flat, one per row in df_out)
    ang_pid_flat = np.zeros(len(df_out), dtype=np.int32)
    ang_mcidx_flat = np.zeros(len(df_out), dtype=np.int32)
    ang_qual_flat = np.full(len(df_out), np.nan, dtype=np.float32)
    delta_th_flat = np.full(len(df_out), np.nan, dtype=np.float32)
    delta_phi_flat = np.full(len(df_out), np.nan, dtype=np.float32)
    delta_p_flat = np.full(len(df_out), np.nan, dtype=np.float32)
    ang_match_pass_flat = np.zeros(len(df_out), dtype=np.int32)
    ang_pid_ok_flat = np.zeros(len(df_out), dtype=np.int32)
    ang_truth_like_flat = np.zeros(len(df_out), dtype=np.int32)

    # Get event boundaries from REC arrays
    rec_pid_awk = arrs["REC_Particle_pid"]
    event_sizes = np.array([len(x) for x in rec_pid_awk], dtype=int)
    event_boundaries = np.cumsum([0] + list(event_sizes))

    row_idx = 0
    for ev_idx in range(len(mc_pid_awk)):
        start_row = event_boundaries[ev_idx]
        end_row = event_boundaries[ev_idx + 1]
        n_rec_in_ev = end_row - start_row

        pid_mc_ev = np.array(mc_pid_awk[ev_idx], dtype=int)
        px_mc_ev = np.array(mc_px_awk[ev_idx], dtype=float)
        py_mc_ev = np.array(mc_py_awk[ev_idx], dtype=float)
        pz_mc_ev = np.array(mc_pz_awk[ev_idx], dtype=float)

        for i_local in range(n_rec_in_ev):
            i_global = start_row + i_local

            px_r = float(rec_px[i_global])
            py_r = float(rec_py[i_global])
            pz_r = float(rec_pz[i_global])
            expected_pid = int(expected_mc_pid_arr[i_global])

            p_rec = np.sqrt(px_r**2 + py_r**2 + pz_r**2)
            if p_rec < 0.001:
                continue

            theta_rec = np.arctan2(np.sqrt(px_r**2 + py_r**2), pz_r)
            phi_rec = np.arctan2(py_r, px_r)

            best_quality = 1e6
            best_mc_pid = 0
            best_mc_idx = -1
            best_delta_th = np.nan
            best_delta_phi = np.nan
            best_delta_p = np.nan

            # Loop over MC particles in this event, filter to expected PID
            for i_mc in range(len(pid_mc_ev)):
                pid_m = int(pid_mc_ev[i_mc])

                # Only match to expected MC PID
                if pid_m != expected_pid:
                    continue

                px_m = float(px_mc_ev[i_mc])
                py_m = float(py_mc_ev[i_mc])
                pz_m = float(pz_mc_ev[i_mc])

                p_mc = np.sqrt(px_m**2 + py_m**2 + pz_m**2)
                if p_mc < 0.001:
                    continue

                theta_mc = np.arctan2(np.sqrt(px_m**2 + py_m**2), pz_m)
                phi_mc = np.arctan2(py_m, px_m)

                # Angular differences
                delta_theta = abs(float(theta_rec - theta_mc)) * 180.0 / np.pi
                delta_phi = abs(float(phi_rec - phi_mc)) * 180.0 / np.pi

                # Wrap phi difference (max 180°)
                if delta_phi > 180.0:
                    delta_phi = 360.0 - delta_phi

                # Momentum difference
                delta_p_rel = abs(p_rec - p_mc) / p_rec if p_rec > 0 else 1e6

                # Quality: weighted angular (RichCap style)
                quality = (delta_theta / 6.0) + (delta_phi / 10.0)

                # Keep best match
                if quality < best_quality:
                    best_quality = quality
                    best_mc_pid = pid_m
                    best_mc_idx = i_mc
                    best_delta_th = delta_theta
                    best_delta_phi = delta_phi
                    best_delta_p = delta_p_rel

            # Store results
            ang_pid_flat[i_global] = best_mc_pid
            ang_mcidx_flat[i_global] = best_mc_idx
            ang_qual_flat[i_global] = float(best_quality) if best_quality < 1e6 else np.nan
            delta_th_flat[i_global] = best_delta_th
            delta_phi_flat[i_global] = best_delta_phi
            delta_p_flat[i_global] = best_delta_p

            # ang_match_pass: both angle cuts satisfied (RichCap thresholds)
            is_pass = False
            if not np.isnan(best_delta_th) and not np.isnan(best_delta_phi):
                if best_delta_th < 6.0 and best_delta_phi < 10.0:
                    ang_match_pass_flat[i_global] = 1
                    is_pass = True

            # ang_pid_ok: matched to expected PID
            is_pid_ok = False
            if best_mc_pid == expected_pid:
                ang_pid_ok_flat[i_global] = 1
                is_pid_ok = True

            # ang_truth_like: both pass AND pid_ok
            if is_pass and is_pid_ok:
                ang_truth_like_flat[i_global] = 1

    # Add columns to output
    df_out["ang_match_pid"] = ang_pid_flat
    df_out["ang_match_mcindex"] = ang_mcidx_flat
    df_out["ang_match_quality"] = ang_qual_flat
    df_out["delta_theta"] = delta_th_flat
    df_out["delta_phi"] = delta_phi_flat
    df_out["delta_p_rel"] = delta_p_flat
    df_out["ang_match_pass"] = ang_match_pass_flat
    df_out["ang_pid_ok"] = ang_pid_ok_flat
    df_out["ang_truth_like"] = ang_truth_like_flat

    # Add agreement flag between COATJAVA and angular matches
    # Prefer mcindex comparison (more reliable than PID)
    if "mcindex_raw" in df_out.columns:
        df_out["match_agreement"] = (df_out["mcindex_raw"] == df_out["ang_match_mcindex"]).astype(int)
    elif "mc_pid_raw" in df_out.columns:
        df_out["match_agreement"] = (df_out["mc_pid_raw"] == df_out["ang_match_pid"]).astype(int)

    return df_out
