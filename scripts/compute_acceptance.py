#!/usr/bin/env python3
"""
compute_acceptance.py

Computes per-bin acceptance correction from reco and gen yield CSVs
produced by count_sidis_bins.py.

═══════════════════════════════════════════════════════════════════════════════
 WHERE THIS FILE FITS IN THE PIPELINE
═══════════════════════════════════════════════════════════════════════════════
  ROOT files
      → root_2_parquet.py         parquet files (no physics cuts)
      → count_sidis_bins.py       binned yield CSVs (data, reco, gen)
      → compute_acceptance.py     acceptance correction CSV  ← THIS FILE
      → notebooks 04/05/06/07     physics extraction

  Supporting scripts:
      physics_constants.py  — beam energy, particle masses
      common_cuts.py        — detector geometry helpers
      electron_cuts.py      — electron PID cuts
      pip_cuts.py           — pion PID cuts
      analysis_cuts.py      — DIS/SIDIS kinematic cut sets  ← CHANGE CUTS HERE
      count_sidis_bins.py   — parquet → binned CSV
      compute_acceptance.py — reco+gen CSV → acceptance CSV ← THIS FILE

═══════════════════════════════════════════════════════════════════════════════
 HOW TO USE
═══════════════════════════════════════════════════════════════════════════════
  # LD2 (run once after count_sidis_bins.py)
  python3 scripts/compute_acceptance.py \
      --reco /path/to/yields/reco_LD2.csv \
      --gen  /path/to/yields/gen_LD2.csv  \
      --out  /path/to/corrections/acceptance_LD2.csv

  # When C, Cu, Sn are available — same command, just change files:
  python3 scripts/compute_acceptance.py \
      --reco /path/to/yields/reco_C.csv \
      --gen  /path/to/yields/gen_C.csv  \
      --out  /path/to/corrections/acceptance_C.csv

  # On iFarm — same command, change paths:
  python3 scripts/compute_acceptance.py \
      --reco /volatile/clas12/rg-d/<user>/yields/reco_LD2.csv \
      --gen  /volatile/clas12/rg-d/<user>/yields/gen_LD2.csv  \
      --out  /volatile/clas12/rg-d/<user>/corrections/acceptance_LD2.csv

═══════════════════════════════════════════════════════════════════════════════
 ACCEPTANCE FORMULA
═══════════════════════════════════════════════════════════════════════════════
  A(bin) = M_reco / M_gen
         = (N_pip_reco / N_e_reco) / (N_pip_gen / N_e_gen)

  Multiplicity-based acceptance — correctly accounts for BOTH:
    - Pion detection efficiency   (N_pip_reco / N_pip_gen)
    - Electron detection efficiency (N_e_reco / N_e_gen)

  This ensures that R = M_data / A is properly normalized:
    R = (N_pip_data/N_e_data) / [(N_pip_reco/N_e_reco) / (N_pip_gen/N_e_gen)]

  Statistical uncertainty (error propagation):
    dA/A = sqrt((dM_reco/M_reco)^2 + (dM_gen/M_gen)^2)
    where dM/M = sqrt(1/N_pip + 1/N_e)

═══════════════════════════════════════════════════════════════════════════════
 OUTPUT CSV COLUMNS
═══════════════════════════════════════════════════════════════════════════════
  {dim}_lo, {dim}_hi   bin edges for each axis dimension
  {dim}_mean           bin mean (from reco)
  N_pip_reco, N_e_reco   reco pion and electron counts in this bin
  N_pip_gen,  N_e_gen    gen  pion and electron counts in this bin
  A_pi, dA_pi            pion-only acceptance = N_pip_reco/N_pip_gen
                           USE FOR: pT broadening, cos φ, cos 2φ, sin φ BSA
  A_e,  dA_e             electron-only acceptance = N_e_reco/N_e_gen
                           USE FOR: DIS denominator correction
  NOTE: For R(zh), compute A = A_pi/A_e in the notebook

═══════════════════════════════════════════════════════════════════════════════
 WHICH NOTEBOOKS USE THIS OUTPUT
═══════════════════════════════════════════════════════════════════════════════
  Notebook 04  — R(zh): divides M_data by A per zh bin
  Notebook 05  — ΔpT²: corrects mean_pT2 per bin
  Notebook 06  — cos φ, cos 2φ: corrects φh distribution per (zh, pT2) bin
  Notebook 07  — sin φ BSA: acceptance largely cancels (same for N+/N-)
"""

import argparse
import os
import numpy as np
import pandas as pd


def get_dims(df):
    """Infer axis dimension names from _lo column names."""
    return [c[:-3] for c in df.columns if c.endswith('_lo')]


def main():
    ap = argparse.ArgumentParser(
        description="Compute acceptance correction: reco + gen CSVs → acceptance CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--reco", required=True,
                    help="Reco yield CSV produced by count_sidis_bins.py")
    ap.add_argument("--gen",  required=True,
                    help="Gen yield CSV produced by count_sidis_bins.py")
    ap.add_argument("--out",  required=True,
                    help="Output acceptance CSV path (e.g. corrections/acceptance_LD2.csv)")
    args = ap.parse_args()

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    reco = pd.read_csv(args.reco)
    gen  = pd.read_csv(args.gen)
    print(f"Reco: {len(reco)} bins  |  Gen: {len(gen)} bins")

    # ── Identify bin dimensions ────────────────────────────────────────────────
    dims     = get_dims(reco)
    bin_lo   = [f"{d}_lo"   for d in dims]
    bin_hi   = [f"{d}_hi"   for d in dims]
    bin_mean = [f"{d}_mean" for d in dims]
    merge_on = bin_lo + bin_hi
    print(f"Dimensions: {dims}")

    # ── Merge reco and gen on bin edges ───────────────────────────────────────
    # Must have identical binning — enforced by using same --bins in count_sidis_bins.py
    reco_sel = reco[merge_on + bin_mean + ['N_pip', 'N_e']].rename(
        columns={'N_pip': 'N_pip_reco', 'N_e': 'N_e_reco'})
    gen_sel  = gen[merge_on + ['N_pip', 'N_e']].rename(
        columns={'N_pip': 'N_pip_gen', 'N_e': 'N_e_gen'})
    df = reco_sel.merge(gen_sel, on=merge_on, how='inner')

    # ── Keep only bins with events in both reco and gen ───────────────────────
    # Bins with zero counts have undefined acceptance → skip
    df = df[
        (df['N_pip_reco'] > 0) & (df['N_pip_gen'] > 0) &
        (df['N_e_reco']   > 0) & (df['N_e_gen']   > 0)
    ].copy()
    print(f"Bins with reco>0 and gen>0: {len(df)}")

    # ── Compute acceptance ────────────────────────────────────────────────────
    # A_pi(bin) = N_pip_reco / N_pip_gen
    #   USE FOR: pT broadening, cos φ, cos 2φ, sin φ BSA  (pion observable)
    df['A_pi']  = df['N_pip_reco'] / df['N_pip_gen']
    df['dA_pi'] = df['A_pi'] * np.sqrt(
        1.0/df['N_pip_reco'] + 1.0/df['N_pip_gen'])

    # A_e(bin)  = N_e_reco / N_e_gen
    #   USE FOR: DIS electron denominator correction
    df['A_e']  = df['N_e_reco'] / df['N_e_gen']
    df['dA_e'] = df['A_e'] * np.sqrt(
        1.0/df['N_e_reco'] + 1.0/df['N_e_gen'])

    # NOTE: For R(zh) multiplicity ratio, compute in the notebook:
    #   A    = A_pi / A_e
    #   dA/A = sqrt((dA_pi/A_pi)^2 + (dA_e/A_e)^2)

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_cols = (merge_on + bin_mean +
                ['N_pip_reco', 'N_e_reco', 'N_pip_gen', 'N_e_gen',
                 'A_pi', 'dA_pi',   # pT broad / φ modulations / BSA
                 'A_e',  'dA_e'])   # electron denominator
    df[out_cols].to_csv(args.out, index=False)
    print(f"Wrote {len(df)} bins → {args.out}")

    # ── Summary: acceptance integrated over non-zh dimensions ─────────────────
    if 'zh_lo' in df.columns:
        print("\nAcceptance per zh bin (summed over other dimensions):")
        zh_grp = df.groupby(['zh_lo', 'zh_hi'], as_index=False).agg(
            N_pip_reco=('N_pip_reco', 'sum'),
            N_pip_gen =('N_pip_gen',  'sum'),
            N_e_reco  =('N_e_reco',   'first'),
            N_e_gen   =('N_e_gen',    'first'),
        )
        zh_grp['A_pi'] = zh_grp['N_pip_reco'] / zh_grp['N_pip_gen']
        zh_grp['A_e']  = zh_grp['N_e_reco']   / zh_grp['N_e_gen']
        zh_grp['A']    = zh_grp['A_pi'] / zh_grp['A_e']   # for R(zh)
        print(zh_grp[['zh_lo','zh_hi','A_pi','A_e','A']].to_string(index=False))


if __name__ == "__main__":
    main()
