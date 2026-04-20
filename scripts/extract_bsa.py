#!/usr/bin/env python3
"""
extract_bsa.py

Extracts the sin φh beam spin asymmetry (BSA) from binned yield CSVs
produced by count_sidis_bins.py.

═══════════════════════════════════════════════════════════════════════════════
 WHERE THIS FILE FITS IN THE PIPELINE
═══════════════════════════════════════════════════════════════════════════════
  ROOT files
      → root_2_parquet.py          parquet files
      → count_sidis_bins.py        binned yield CSVs  (must include phi_h axis)
      → extract_bsa.py             BSA results CSV    ← THIS FILE
      → notebook 07_bsa_sinphi     plots only

═══════════════════════════════════════════════════════════════════════════════
 HOW TO USE
═══════════════════════════════════════════════════════════════════════════════
  # Step 1: produce CSV with phi_h axis (any other axes allowed)
  python3 scripts/count_sidis_bins.py data*.parquet \
      --axes zh,phi_h --apply-sidis \
      --out-csv yields/data_LD2_bsa.csv

  # Step 2: extract BSA (no acceptance correction needed for preliminary)
  python3 scripts/extract_bsa.py \
      --csv  yields/data_LD2_bsa.csv \
      --pbeam 0.85 \
      --out  results/bsa_LD2.csv

  # Step 3: with acceptance correction (optional)
  python3 scripts/extract_bsa.py \
      --csv  yields/data_LD2_bsa.csv \
      --acc  yields/acceptance_LD2.csv \
      --pbeam 0.85 \
      --out  results/bsa_LD2_ac.csv

  # Works for ANY axis combination — auto-detected from the CSV:
  #   --axes zh,phi_h             → A_LU^sinφ vs zh
  #   --axes zh,pT2,phi_h         → A_LU^sinφ vs (zh, pT2)
  #   --axes Q2,xB,zh,pT2,phi_h  → full 5D

═══════════════════════════════════════════════════════════════════════════════
 PHYSICS
═══════════════════════════════════════════════════════════════════════════════
  Raw asymmetry per φh bin:
      A_LU(φh) = (N+ - N-) / (N+ + N-) / P_beam

  Fit model (per kinematic bin):
      A_LU(φh) = a1*sin(φh) + a2*sin(2φh)
      a1 = A_LU^sinφ   ← main physics observable
      a2 = A_LU^sin2φ  ← higher twist

  Acceptance correction (optional):
      A_LU(φh) = (N+/A_pi - N-/A_pi) / (N+/A_pi + N-/A_pi)
               = (N+ - N-) / (N+ + N-)   ← A_pi cancels exactly
      So acceptance correction has NO effect on BSA to first order.
      Only residual φh-dependent efficiency differences between helicities
      would matter — negligible for CLAS12.

═══════════════════════════════════════════════════════════════════════════════
 OUTPUT CSV COLUMNS
═══════════════════════════════════════════════════════════════════════════════
  {dim}_lo, {dim}_hi, {dim}_mean   bin edges and mean for each non-phi axis
  N_plus_tot, N_minus_tot          total helicity counts in this kinematic bin
  A_LU_sinphi,  dA_LU_sinphi      sin φh moment and statistical uncertainty
  A_LU_sin2phi, dA_LU_sin2phi     sin 2φh moment and statistical uncertainty
  chi2_ndf                         fit quality
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ── Fit model ─────────────────────────────────────────────────────────────────
def bsa_model(phi_rad, a1, a2):
    """A_LU(φh) = a1*sin(φh) + a2*sin(2φh)"""
    return a1 * np.sin(phi_rad) + a2 * np.sin(2 * phi_rad)


def get_dims(df):
    """Infer axis dimension names from _lo column names."""
    return [c[:-3] for c in df.columns if c.endswith('_lo')]


def extract_bsa_bin(grp, p_beam):
    """
    Fit BSA for one kinematic bin (all phi_h rows belonging to it).
    Returns dict of fit results, or None if insufficient data.
    """
    grp = grp[(grp['N_plus'] > 0) & (grp['N_minus'] > 0)].copy()
    if len(grp) < 4:
        return None

    phi_deg = (grp['phi_h_lo'] + grp['phi_h_hi']) / 2.0
    phi_rad = np.deg2rad(phi_deg).to_numpy()

    N_plus  = grp['N_plus'].to_numpy()
    N_minus = grp['N_minus'].to_numpy()
    N_tot   = N_plus + N_minus

    # Raw asymmetry and binomial statistical error
    A_raw  = (N_plus - N_minus) / N_tot
    dA_raw = np.sqrt(np.maximum(1.0 - A_raw**2, 0.0) / N_tot)

    # Divide by beam polarization
    A_lu  = A_raw  / p_beam
    dA_lu = dA_raw / p_beam

    try:
        popt, pcov = curve_fit(
            bsa_model, phi_rad, A_lu,
            sigma=dA_lu, absolute_sigma=True,
            p0=[0.0, 0.0], maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
    except (RuntimeError, ValueError):
        return None

    # Fit quality
    residuals = A_lu - bsa_model(phi_rad, *popt)
    chi2 = np.sum((residuals / dA_lu)**2)
    ndf  = len(phi_rad) - 2
    chi2_ndf = chi2 / ndf if ndf > 0 else np.nan

    return {
        'N_plus_tot':     N_plus.sum(),
        'N_minus_tot':    N_minus.sum(),
        'A_LU_sinphi':    popt[0],
        'dA_LU_sinphi':   perr[0],
        'A_LU_sin2phi':   popt[1],
        'dA_LU_sin2phi':  perr[1],
        'chi2_ndf':       chi2_ndf,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Extract sin φh BSA from binned yield CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv",    required=True,
                    help="Data yield CSV from count_sidis_bins.py (must include phi_h axis)")
    ap.add_argument("--acc",    default=None,
                    help="Acceptance CSV from compute_acceptance.py (optional — A_pi cancels in BSA)")
    ap.add_argument("--pbeam",  type=float, default=0.85,
                    help="Beam polarization (default: 0.85 — update with run-averaged value)")
    ap.add_argument("--out",    required=True,
                    help="Output results CSV path  e.g. results/bsa_LD2.csv")
    args = ap.parse_args()

    # ── Load data CSV ──────────────────────────────────────────────────────────
    data = pd.read_csv(args.csv)
    print(f"Loaded {len(data)} bins from {args.csv}")

    if 'phi_h_lo' not in data.columns:
        sys.exit("ERROR: phi_h axis not found. Rerun count_sidis_bins.py with --axes ...,phi_h")

    if 'N_plus' not in data.columns or 'N_minus' not in data.columns:
        sys.exit("ERROR: N_plus/N_minus columns not found. Check count_sidis_bins.py output.")

    # ── Auto-detect dimensions ─────────────────────────────────────────────────
    all_dims   = get_dims(data)
    group_dims = [d for d in all_dims if d != 'phi_h']
    group_lo   = [f'{d}_lo' for d in group_dims]
    group_hi   = [f'{d}_hi' for d in group_dims]
    group_keys = group_lo + group_hi
    print(f"Dimensions detected : {all_dims}")
    print(f"Grouping by         : {group_dims}")
    print(f"Fitting φh within each group | P_beam = {args.pbeam}")

    # ── Optional: load acceptance (informational — A_pi cancels in BSA) ────────
    if args.acc:
        acc = pd.read_csv(args.acc)
        print(f"Acceptance CSV loaded ({len(acc)} bins) — note: A_pi cancels in BSA ratio")

    # ── Fit BSA per kinematic bin ──────────────────────────────────────────────
    rows = []
    for keys, grp in data.groupby(group_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)

        fit = extract_bsa_bin(grp, args.pbeam)
        if fit is None:
            print(f"  Skipped (insufficient phi bins): {dict(zip(group_keys, keys))}")
            continue

        row = {}
        n = len(group_dims)
        for i, dim in enumerate(group_dims):
            row[f'{dim}_lo']   = keys[i]
            row[f'{dim}_hi']   = keys[i + n]
            row[f'{dim}_mean'] = grp[f'{dim}_mean'].iloc[0]

        row.update(fit)
        rows.append(row)

    if not rows:
        sys.exit("ERROR: No bins fitted. Check that N_plus/N_minus are non-zero.")

    df_out = pd.DataFrame(rows)
    print(f"\nFitted {len(df_out)} kinematic bins")
    print(df_out[[c for c in df_out.columns if 'mean' in c or 'A_LU' in c]].to_string(index=False))

    # ── Write output ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"\nWrote {len(df_out)} bins → {args.out}")


if __name__ == "__main__":
    main()
