# truth_matching.py

"""
Truth matching helpers for CLAS12 MC samples.

Implements the REC::Particle -> MC::Particle matching using the MC::RecMatch bank:

For each reconstructed particle (REC::Particle index iRec):

    quality = MC_RecMatch_quality[row where pindex == iRec]
    mcind   = MC_RecMatch_mcindex[same row]
    mc_pid  = MC_Particle_pid[mcind]

We then:
  - add 'mc_pid' and 'match_quality' columns to df_all
  - optionally cut on match_quality >= quality_min

Usage:
  - Only for simulation samples that have MC_* and MC_RecMatch_* branches.
  - Called in run_single_sample.py before electron_cuts for sim.
"""

import numpy as np
import awkward as ak


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
           - mc_pid
           - match_quality
        and possibly filtered by quality_min.
    """

    # Event-wise arrays
    rec_pid    = arrs["REC_Particle_pid"]          # just for event sizes
    rm_pindex  = arrs["MC_RecMatch_pindex"]
    rm_mcindex = arrs["MC_RecMatch_mcindex"]
    rm_quality = arrs["MC_RecMatch_quality"]
    mc_pid_arr = arrs["MC_Particle_pid"]

    mc_pid_per_event        = []
    match_quality_per_event = []

    # Loop over events (one "row" per event in each awkward array)
    for pid_ev, pidx_ev, mcidx_ev, qual_ev, mc_pid_ev in zip(
        rec_pid, rm_pindex, rm_mcindex, rm_quality, mc_pid_arr
    ):
        n_part = len(pid_ev)

        # Default: no match
        mc_pid_evt   = np.zeros(n_part, dtype=np.int32)
        qual_evt_out = np.zeros(n_part, dtype=np.float32)

        # Loop over rows of MC::RecMatch for this event
        # pidx_ev[i]: REC::Particle index (iRec)
        # mcidx_ev[i]: MC::Particle index
        # qual_ev[i]: match quality
        for pidx, mcind, q in zip(pidx_ev, mcidx_ev, qual_ev):
            if pidx < 0 or pidx >= n_part:
                continue

            # attach quality
            qual_evt_out[pidx] = q

            # attach MC pid if mcind is valid
            if 0 <= mcind < len(mc_pid_ev):
                mc_pid_evt[pidx] = mc_pid_ev[mcind]
            else:
                mc_pid_evt[pidx] = 0  # no valid match

        mc_pid_per_event.append(mc_pid_evt)
        match_quality_per_event.append(qual_evt_out)

    # Flatten to match flat REC_Particle ordering used in df_all
    mc_pid_flat        = ak.to_numpy(ak.flatten(ak.Array(mc_pid_per_event)))
    match_quality_flat = ak.to_numpy(ak.flatten(ak.Array(match_quality_per_event)))

    if mc_pid_flat.shape[0] != len(df_all):
        raise RuntimeError(
            f"Truth-matching length mismatch: mc_pid_flat={mc_pid_flat.shape[0]}, df_all={len(df_all)}"
        )

    df_all = df_all.copy()
    df_all["mc_pid"] = mc_pid_flat
    df_all["match_quality"] = match_quality_flat

    # Optional quality cut (like requiring quality > 0.98 in the C++ snippet)
    if quality_min is not None:
        mask_q = df_all["match_quality"].to_numpy() >= quality_min
        df_all = df_all[mask_q].copy()

    return df_all


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
