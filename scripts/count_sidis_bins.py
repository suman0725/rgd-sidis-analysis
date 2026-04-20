#!/usr/bin/env python3
"""
count_sidis_bins.py
High-performance, memory-efficient SIDIS binning with Mean-Tracking.

═══════════════════════════════════════════════════════════════════════════════
 WHERE THIS FILE FITS IN THE PIPELINE
═══════════════════════════════════════════════════════════════════════════════
  ROOT files
      → root_2_parquet.py       produces parquet files (no physics cuts)
      → count_sidis_bins.py     bins parquet → kinematic yields CSV  ← THIS FILE
      → notebooks 04/05/06      physics extraction (R, pT broadening, moments)

  Supporting scripts (do NOT edit unless you know what you are changing):
      physics_constants.py  — beam energy, particle masses
      common_cuts.py        — detector geometry helpers (is_fd, polarity)
      electron_cuts.py      — electron detector and PID cuts
      pip_cuts.py           — pion detector and PID cuts
      analysis_cuts.py      — DIS/SIDIS kinematic cut sets  ← CHANGE CUTS HERE

═══════════════════════════════════════════════════════════════════════════════
 INSTALLATION (first time only, on any machine including iFarm)
═══════════════════════════════════════════════════════════════════════════════
  python3 -m pip install --user boost-histogram pandas pyarrow

═══════════════════════════════════════════════════════════════════════════════
 HOW TO USE
═══════════════════════════════════════════════════════════════════════════════
  Basic (1D in zh, full SIDIS cuts):
    python3 count_sidis_bins.py /path/to/parquet/data_*.parquet \
        --axes zh --apply-sidis --out-csv yields/data_LD2_zh.csv

  2D (zh and pT2):
    python3 count_sidis_bins.py /path/to/parquet/data_*.parquet \
        --axes zh,pT2 --apply-sidis --out-csv yields/data_LD2_zh_pT2.csv

  For azimuthal moments — include phi_h as axis:
    python3 count_sidis_bins.py /path/to/parquet/data_*.parquet \
        --axes zh,pT2,phi_h --apply-sidis --out-csv yields/data_LD2_phi.csv

  On iFarm — same command, just change the parquet path:
    python3 count_sidis_bins.py /volatile/clas12/rg-d/<user>/parquet/*.parquet \
        --axes zh --apply-sidis --out-csv yields/data_LD2_zh.csv

═══════════════════════════════════════════════════════════════════════════════
 HOW TO SWITCH CUT SETS  ← CHANGE ONLY THIS
═══════════════════════════════════════════════════════════════════════════════
  Open analysis_cuts.py and change ONE line:
      CUT_SET = 'standard'   →   CUT_SET = 'tight'
  Options: 'loose', 'standard' (Daniel's RG-D cuts), 'tight'
  This script automatically picks up the new values — no changes needed here.

  To override a single cut on the command line (overrides analysis_cuts.py):
      --q2-min 1.5   --zh-min 0.25   etc.

═══════════════════════════════════════════════════════════════════════════════
 CUT MODES (match notebook 01/02/03 toggle flags)
═══════════════════════════════════════════════════════════════════════════════
  No flags      → raw counts, no physics cuts (full kinematics)
  --apply-dis   → DIS electron cuts only: Q2, W, y
  --apply-sidis → DIS electron cuts + SIDIS hadron cuts: zh, pT2
                  (--apply-sidis automatically implies --apply-dis)

═══════════════════════════════════════════════════════════════════════════════
 OUTPUT CSV COLUMNS
═══════════════════════════════════════════════════════════════════════════════
  {dim}_lo, {dim}_hi   bin edges for each axis dimension
  {dim}_mean           true mean of variable within the bin (not bin centre)
  N_e                  number of DIS electrons (denominator for multiplicity)
  V_e                  variance of N_e (for error propagation)
  N_pip                number of SIDIS pions (numerator for multiplicity)
  V_pip                variance of N_pip
  N_plus, V_plus       pion counts for positive helicity beam (for BSA)
  N_minus, V_minus     pion counts for negative helicity beam (for BSA)
  mean_pT2, err_pT2    mean pT² and its error per bin (for pT broadening)

═══════════════════════════════════════════════════════════════════════════════
 WHAT NOTEBOOKS USE THIS OUTPUT
═══════════════════════════════════════════════════════════════════════════════
  Notebook 04  — R (multiplicity ratio A/LD2) and pT broadening
                 uses: N_pip, N_e, mean_pT2 for data + reco + gen
  Notebook 05  — cos phi, cos 2phi moments
                 uses: N_pip binned in phi_h → fit per (zh, pT2) bin
  Notebook 06  — sin phi beam spin asymmetry (BSA)
                 uses: N_plus, N_minus per bin
"""

import argparse
import sys
import glob
import os
from itertools import product
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import boost_histogram as bh
except ImportError:
    sys.stderr.write("boost-histogram is required. Install with: python3 -m pip install --user boost-histogram\n")
    raise

# ── Import cut defaults from analysis_cuts.py ────────────────────────────────
# Automatically finds analysis_cuts.py in the same scripts/ folder.
# Change CUT_SET there — values propagate here without touching this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_cuts import Q2_MIN, W_MIN, Y_MIN, Y_MAX, ZH_MIN, ZH_MAX, PT2_MAX, CUT_SET

# ── Default kinematic bin edges ───────────────────────────────────────────────
# Used when --bins is not specified on the command line.
# Override specific dimensions with: --bins "zh=0,0.2,0.4,0.6;pT2=0,0.5,1.0,1.5"
DEFAULT_BINS: Dict[str, List[float]] = {
    "Q2":   [1.0, 2.0, 4.0, 8.0],                               # GeV^2
    "xB":   [0.0, 0.2, 0.4, 0.6, 1.1],                          # Bjorken x
    "nu":   [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],    # GeV
    "zh":   [0.0, 0.3, 0.5, 0.7, 0.9, 1.1],                     # hadron energy fraction
    "pT2":  [0.0, 0.3, 0.7, 1.2, 2.0, 3.0],                     # GeV^2
    "phi_h":[0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0],     # degrees
}


def parse_binspec(spec: str) -> Dict[str, List[float]]:
    """Parse --bins string like 'zh=0,0.2,0.4;pT2=0,1,2' into a dict of edges."""
    out: Dict[str, List[float]] = {}
    if not spec: return out
    for part in [p.strip() for p in spec.split(";") if p.strip()]:
        if "=" not in part: continue
        dim, vals = part.split("=", 1)
        edges = [float(x) for x in vals.split(",") if x.strip()]
        if len(edges) >= 2: out[dim.strip()] = sorted(edges)
    return out


def ensure_bins(axis_list: List[str], user_bins: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Return bin edges for each dimension, falling back to DEFAULT_BINS."""
    bins: Dict[str, List[float]] = {}
    for dim in set(axis_list):
        bins[dim] = user_bins.get(dim, DEFAULT_BINS.get(dim, [0, 1]))
    return bins


def build_axes(dim_order: List[str], bins: Dict[str, List[float]]):
    """Build boost-histogram axes for the given dimensions."""
    axes = []
    for dim in dim_order:
        edges = bins[dim]
        # phi_h gets a circular axis so 0° and 360° are treated as the same boundary
        if dim == "phi_h" and len(edges) >= 2 and edges[0] == 0.0 and edges[-1] == 360.0:
            axes.append(bh.axis.Regular(len(edges) - 1, edges[0], edges[-1], circular=True))
            continue
        axes.append(bh.axis.Variable(edges))
    return axes


def main():
    ap = argparse.ArgumentParser(
        description="SIDIS bin counter: parquet files → kinematic yields CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input / output ────────────────────────────────────────────────────────
    ap.add_argument("parquet", nargs='+',
                    help="Parquet file(s) to process. Wildcards supported.")
    ap.add_argument("--out-csv",
                    help="Output CSV path. e.g. yields/data_LD2_zh.csv")

    # ── Binning ───────────────────────────────────────────────────────────────
    ap.add_argument("--axes", default="zh",
                    help="Comma-separated hadron axes. e.g. zh  or  zh,pT2  or  zh,pT2,phi_h")
    ap.add_argument("--bins",
                    help='Custom bin edges. e.g. "zh=0,0.3,0.5,0.7;pT2=0,0.5,1.0,1.5"')
    ap.add_argument("--integrate-over", default="Q2,xB",
                    help="Dimensions to sum over when computing electron normalisation.")

    # ── Cut mode flags ────────────────────────────────────────────────────────
    # Change CUT_SET in analysis_cuts.py to switch between loose/standard/tight.
    # Use the CLI flags below to toggle which cuts are applied.
    ap.add_argument("--apply-dis",   action="store_true",
                    help="Apply DIS cuts to select electrons: Q2, W, y")
    ap.add_argument("--apply-sidis", action="store_true",
                    help="Apply SIDIS cuts to select hadrons: zh, pT2 (implies --apply-dis)")

    # ── Cut thresholds ────────────────────────────────────────────────────────
    # Defaults loaded from analysis_cuts.py. Override on CLI if needed.
    # To change defaults permanently: edit CUT_SET in analysis_cuts.py instead.
    ap.add_argument("--q2-min", type=float, default=Q2_MIN,
                    help=f"Min Q2 GeV^2  [analysis_cuts.py default: {Q2_MIN}]")
    ap.add_argument("--w-min",  type=float, default=W_MIN,
                    help=f"Min W GeV     [analysis_cuts.py default: {W_MIN}]")
    ap.add_argument("--y-min",  type=float, default=Y_MIN,
                    help=f"Min y         [analysis_cuts.py default: {Y_MIN}]")
    ap.add_argument("--y-max",  type=float, default=Y_MAX,
                    help=f"Max y         [analysis_cuts.py default: {Y_MAX}]")
    ap.add_argument("--zh-min", type=float, default=ZH_MIN,
                    help=f"Min zh        [analysis_cuts.py default: {ZH_MIN}]")
    ap.add_argument("--zh-max", type=float, default=ZH_MAX,
                    help=f"Max zh        [analysis_cuts.py default: {ZH_MAX}]")
    ap.add_argument("--pt2-max",type=float, default=PT2_MAX,
                    help=f"Max pT2 GeV^2 [analysis_cuts.py default: {PT2_MAX}]")

    args = ap.parse_args()

    # ── Collect and validate input files ─────────────────────────────────────
    file_list = sorted(set(f for p in args.parquet for f in glob.glob(p)))
    if not file_list:
        print(f"Error: No files found matching: {args.parquet}"); sys.exit(1)

    # ── Setup axes and bins ───────────────────────────────────────────────────
    axis_list     = [a.strip() for a in args.axes.split(",") if a.strip()]
    integrate_set = set(d.strip() for d in args.integrate_over.split(",") if d.strip()) \
                    if args.integrate_over else set()
    dims_for_bins = list(set(axis_list) | integrate_set)
    bins          = ensure_bins(dims_for_bins, parse_binspec(args.bins))

    # ── Initialise histograms ─────────────────────────────────────────────────
    # hist_e: electron counts in (Q2, xB) — used as DIS denominator
    hist_e = bh.Histogram(*build_axes(["Q2", "xB"], bins), storage=bh.storage.Weight())

    # hist_h: pion counts in user-specified axes — SIDIS numerator
    h_axes = build_axes(axis_list, bins)
    hist_h = bh.Histogram(*h_axes, storage=bh.storage.Weight())

    # mean_trackers: accumulate sum(value × weight) per bin
    # dividing by N_pip later gives the true mean of each variable within the bin
    mean_trackers = {dim: bh.Histogram(*h_axes, storage=bh.storage.Double()) for dim in axis_list}

    # hel_yields: helicity-split pion counts → used for BSA (sin phi)
    hel_yields = {
        "N_plus":  bh.Histogram(*h_axes, storage=bh.storage.Weight()),
        "N_minus": bh.Histogram(*h_axes, storage=bh.storage.Weight()),
    }

    # pt2_moments: accumulate sum(pT2 × w) and sum(pT2^2 × w) for pT broadening
    # mean_pT2 = sum(pT2 × w) / sum(w),  err from variance formula
    pt2_moments = {
        "pt2":    bh.Histogram(*h_axes, storage=bh.storage.Double()),
        "pt2_sq": bh.Histogram(*h_axes, storage=bh.storage.Double()),
    }

    # ── Print active configuration ────────────────────────────────────────────
    print(f"Active cut set : '{CUT_SET}' (from analysis_cuts.py)")
    print(f"  DIS  cuts (electrons) : apply={args.apply_dis or args.apply_sidis}"
          f"  Q2>{args.q2_min}, W>{args.w_min}, {args.y_min}<y<{args.y_max}")
    print(f"  SIDIS cuts (hadrons)  : apply={args.apply_sidis}"
          f"  {args.zh_min}<zh<{args.zh_max}, pT2<{args.pt2_max}")
    print(f"  Axes : {axis_list}")
    print(f"Processing {len(file_list)} files...")

    # ── Detect gen-level files (no detector weights) ─────────────────────────
    # Gen parquet files contain only truth kinematics — no w_e, w_pip, run,
    # or helicity columns. Every row is a valid SIDIS event by construction.
    first_df   = pd.read_parquet(file_list[0])
    is_gen     = "w_e" not in first_df.columns
    if is_gen:
        print("  Detected GEN-level files (no w_e/w_pip columns) — using unit weights")

    # ── Main loop: fill histograms file by file ───────────────────────────────
    for i, fpath in enumerate(file_list):
        df = pd.read_parquet(fpath)

        if is_gen:
            # ── Gen mode: apply kinematic cuts directly, unit weights ─────────
            apply_dis = args.apply_dis or args.apply_sidis
            if apply_dis:
                df = df[
                    (df["Q2"] >= args.q2_min)
                    & (df["W"]  >= args.w_min)
                    & (df["y"]  >= args.y_min)
                    & (df["y"]  <= args.y_max)
                ]
            if args.apply_sidis:
                df = df[
                    (df["zh"]  >= args.zh_min)
                    & (df["zh"]  <= args.zh_max)
                    & (df["pT2"] <= args.pt2_max)
                ]
            # Each row = one pion; each unique event = one electron
            df_e = df.drop_duplicates(subset=["sel_event_idx"]).dropna(subset=["Q2", "xB"])
            hist_e.fill(df_e["Q2"].to_numpy(), df_e["xB"].to_numpy())
            df_h = df.dropna(subset=axis_list)
            df["hel_sign"] = np.nan   # no helicity in gen

        else:
            # ── Step 1: Electron selection — DIS cuts ─────────────────────────
            # Select events containing at least one good DIS electron.
            # --apply-sidis automatically triggers this step (SIDIS requires DIS first).
            apply_dis = args.apply_dis or args.apply_sidis
            if apply_dis:
                e_mask = (
                    (df["w_e"] == 1)
                    & (df["Q2"] >= args.q2_min)
                    & (df["W"]  >= args.w_min)
                    & (df["y"]  >= args.y_min)
                    & (df["y"]  <= args.y_max)
                )
                # Keep only rows belonging to events with a good DIS electron
                passing_events = df.loc[e_mask, "sel_event_idx"].unique()
                df = df[df["sel_event_idx"].isin(passing_events)]

            # Encode helicity: +1, -1, or NaN (unknown/unpolarised)
            df["hel_sign"] = np.where(df["helicity"] != 0,
                                      df["helicity"].astype(float), np.nan)

            # Fill electron histogram (one entry per event, deduplicated)
            df_e = (df[df["w_e"] == 1]
                    .drop_duplicates(subset=["run", "sel_event_idx"])
                    .dropna(subset=["Q2", "xB"]))
            hist_e.fill(df_e["Q2"].to_numpy(), df_e["xB"].to_numpy(),
                        weight=df_e["w_e"].to_numpy())

        # ── Step 2: Hadron selection — SIDIS cuts ─────────────────────────────
        # Gen files: df_h already set above with kinematic cuts + unit weights.
        # Data/Reco files: filter by w_pip==1 and optionally zh, pT2 cuts.
        if not is_gen:
            hadron_mask = (df["w_pip"] == 1)
            if args.apply_sidis:
                hadron_mask = (
                    hadron_mask
                    & (df["zh"]  >= args.zh_min)
                    & (df["zh"]  <= args.zh_max)
                    & (df["pT2"] <= args.pt2_max)
                )
            df_h = df[hadron_mask].dropna(subset=axis_list)

        if not df_h.empty:
            h_data    = [df_h[dim].to_numpy() for dim in axis_list]
            h_weights = np.ones(len(df_h)) if is_gen else df_h["w_pip"].to_numpy()

            # Total pion counts
            hist_h.fill(*h_data, weight=h_weights)

            # Mean value of each axis variable within the bin
            for dim in axis_list:
                mean_trackers[dim].fill(*h_data,
                                        weight=df_h[dim].to_numpy() * h_weights)

            # Helicity-split counts for BSA (sin phi moment) — skipped for gen
            if "hel_sign" not in df_h.columns:
                df_hel = pd.DataFrame()
            else:
                df_hel = df_h.dropna(subset=["hel_sign"])
            if not df_hel.empty:
                for sign, key in [(1.0, "N_plus"), (-1.0, "N_minus")]:
                    df_s = df_hel[df_hel["hel_sign"] == sign]
                    if not df_s.empty:
                        hel_yields[key].fill(
                            *[df_s[dim].to_numpy() for dim in axis_list],
                            weight=df_s["w_pip"].to_numpy())

            # pT2 moments for pT broadening: sum(pT2 w) and sum(pT2^2 w)
            if "pT2" in df_h.columns:
                pt2_vals = df_h["pT2"].to_numpy()
                pt2_moments["pt2"].fill(*h_data, weight=pt2_vals * h_weights)
                pt2_moments["pt2_sq"].fill(*h_data, weight=(pt2_vals**2) * h_weights)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(file_list)} files")

    # ── Build output table ────────────────────────────────────────────────────
    view_e, var_e = hist_e.view().value, hist_e.view().variance
    view_h, var_h = hist_h.view().value, hist_h.view().variance
    axis_bins     = [bins[d] for d in axis_list]

    # Locate Q2 and xB positions in the hadron axis list (may be None if integrated out)
    e_axes_in_hadron = {
        0: axis_list.index("Q2") if "Q2" in axis_list else None,
        1: axis_list.index("xB") if "xB" in axis_list else None,
    }

    view_p    = hel_yields["N_plus"].view()
    view_m    = hel_yields["N_minus"].view()
    pt2_views = {name: hist.view() for name, hist in pt2_moments.items()}

    def mean_and_err(sum_f, sum_f2, N):
        """Compute mean and its statistical error from weighted sums."""
        if N <= 0: return np.nan, np.nan
        mu  = sum_f / N
        var = max((sum_f2 / N) - mu * mu, 0.0)
        return mu, np.sqrt(var / N)

    print("\nGenerating output table...")
    rows = []
    for idx in product(*[range(len(b) - 1) for b in axis_bins]):

        # ── Electron count for this (Q2, xB) slice ────────────────────────────
        # Handles both cases: Q2/xB are explicit axes, or integrated over.
        cur_e, cur_v = view_e, var_e
        if e_axes_in_hadron[0] is not None:
            cur_e, cur_v = cur_e[idx[e_axes_in_hadron[0]], :], cur_v[idx[e_axes_in_hadron[0]], :]
        else:
            cur_e, cur_v = np.sum(cur_e, axis=0), np.sum(cur_v, axis=0)

        if e_axes_in_hadron[1] is not None:
            n_e, v_e = cur_e[idx[e_axes_in_hadron[1]]], cur_v[idx[e_axes_in_hadron[1]]]
        else:
            n_e, v_e = np.sum(cur_e), np.sum(cur_v)

        if n_e == 0: continue  # skip empty bins

        # ── Hadron count for this bin ─────────────────────────────────────────
        n_h, v_h = view_h[idx], var_h[idx]

        # ── Build output row ──────────────────────────────────────────────────
        row = {}
        for j, dim in enumerate(axis_list):
            row[f"{dim}_lo"]   = axis_bins[j][idx[j]]
            row[f"{dim}_hi"]   = axis_bins[j][idx[j] + 1]
            # True mean within the bin (not bin centre)
            row[f"{dim}_mean"] = (mean_trackers[dim].view()[idx] / n_h
                                  if n_h > 0
                                  else (row[f"{dim}_lo"] + row[f"{dim}_hi"]) / 2)

        row.update({"N_e": n_e, "V_e": v_e, "N_pip": n_h, "V_pip": v_h})

        # Helicity yields (for BSA extraction in notebook 06)
        row["N_plus"]  = view_p[idx].value
        row["V_plus"]  = view_p[idx].variance
        row["N_minus"] = view_m[idx].value
        row["V_minus"] = view_m[idx].variance

        # pT2 mean and error per bin (for pT broadening in notebook 04)
        mean_pt2, err_pt2 = mean_and_err(pt2_views["pt2"][idx],
                                          pt2_views["pt2_sq"][idx], n_h)
        row["mean_pT2"] = mean_pt2
        row["err_pT2"]  = err_pt2

        rows.append(row)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) \
            if os.path.dirname(args.out_csv) else None
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"Wrote {len(rows)} bins → {args.out_csv}")
    else:
        print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
