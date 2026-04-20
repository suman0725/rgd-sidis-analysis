#!/usr/bin/env python3
"""
haprad_rc_input.py
Prepare radiative-correction inputs from SIDIS parquet files.

═══════════════════════════════════════════════════════════════════════════════
 WHERE THIS FILE FITS IN THE PIPELINE
═══════════════════════════════════════════════════════════════════════════════
  parquet files (from root_2_parquet.py)
      → count_sidis_bins.py   yields CSV for physics analysis
      → haprad_rc_input.py            bins pions into RC grid         ← THIS FILE
          ├── centroids_{TARGET}.csv   mean (Q², xB, nu, zh, pT², φ) per bin
          └── phi_fits_{TARGET}.csv    φ-fit parameters per 4-D bin
      → exec_rad-corr_chain.sh  calls HAPRAD (uses nu_mean) → RCFactor.txt
      → apply_rc.py             adds rc_factor column to parquet
      → count_sidis_bins.py     weight=w_pip/rc_factor → RC-corrected yields

═══════════════════════════════════════════════════════════════════════════════
 DESIGN CHOICES (first attempt — bins can be updated)
═══════════════════════════════════════════════════════════════════════════════
  • RC grid uses the SAME (Q², xB) bins as the analysis so RC factors map
    directly onto analysis bins — no weighted averaging needed.
  • zh / pT² / φ bins follow Binning.hxx (RGD standard) for HAPRAD.
  • Centroids output BOTH xB_mean and nu_mean per bin.
      - xB_mean : true mean Bjorken-x from parquet column
      - nu_mean : true mean energy-transfer from parquet column  ← HAPRAD uses this
  • Consistency check: nu_computed = Q²/(2·Mp·xB) is compared with nu (parquet).
      A large discrepancy would indicate a bug in root_2_parquet.py physics.
  • φ convention: parquet stores phi_h in [0°, 360°].
                  RC grid uses [-180°, 180°] → phi_PQ = phi_h − 360 if phi_h > 180.

═══════════════════════════════════════════════════════════════════════════════
 DEFAULT RC GRID
═══════════════════════════════════════════════════════════════════════════════
  Q²   [GeV²] : [1.0, 2.0, 4.0, 8.0]        — matches analysis Q² bins
  xB          : [0.0, 0.2, 0.4, 0.6, 1.1]   — matches analysis xB bins
  zh          : [0.3, 0.5, 0.75, 1.0]        — Binning.hxx
  pT²  [GeV²] : [0.0, 0.5, 1.0, 1.5]        — Binning.hxx
  φ    [deg]  : [-180,-108,-36,36,108,180]    — Binning.hxx (5 bins)

═══════════════════════════════════════════════════════════════════════════════
 USAGE
═══════════════════════════════════════════════════════════════════════════════
  # LD2:
  python3 haprad_rc_input.py \\
      /volatile/clas12/suman/00_RGD_Analysis/data/experimental/parquet/LD2/018420/*.parquet \\
      --target LD2 --out-dir /work/clas12/suman/00_RGD_Analysis/haprad_rc

  # CxC:
  python3 haprad_rc_input.py \\
      /volatile/clas12/suman/00_RGD_Analysis/data/experimental/parquet/CxC/018454/*.parquet \\
      --target CxC --out-dir /work/clas12/suman/00_RGD_Analysis/haprad_rc

  # Override any bin edges:
  python3 haprad_rc_input.py *.parquet --target LD2 \\
      --bins "Q2=1,2,4,8;xB=0,0.2,0.4,0.6,1.1;zh=0.3,0.5,0.75,1.0"
"""

import argparse
import sys
import glob
import os
from itertools import product

import numpy as np
import pandas as pd

try:
    import boost_histogram as bh
except ImportError:
    sys.stderr.write(
        "boost-histogram required.  Install: python3 -m pip install --user boost-histogram\n"
    )
    raise

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    sys.stderr.write(
        "[WARN] scipy not found — phi fits will be skipped.\n"
        "Install: python3 -m pip install --user scipy\n"
    )

# ── Proton mass for nu ↔ xB conversion ───────────────────────────────────────
M_PROTON = 0.9382720813   # GeV  (matches physics_constants.py)

# ── Default RC grid ───────────────────────────────────────────────────────────
# Q² and xB match analysis bins (count_sidis_bins.py defaults).
# zh, pT², φ follow Binning.hxx (RGD standard for HAPRAD).
# Update these for the next analysis iteration — no other file needs to change.
DEFAULT_BINS = {
    "Q2":  [1.0, 2.0, 4.0, 8.0],
    "xB":  [0.0, 0.2, 0.4, 0.6, 1.1],
    "zh":  [0.3, 0.5, 0.75, 1.0],
    "pT2": [0.0, 0.5, 1.0, 1.5],
    "phi": [-180.0, -108.0, -36.0, 36.0, 108.0, 180.0],
}

# ── SIDIS cuts — match analysis_cuts.py 'standard', relaxed at RC grid edges ─
_CUTS = dict(
    Q2_MIN=1.0, W_MIN=2.0, Y_MIN=0.25, Y_MAX=0.85,
    ZH_MIN=0.3, ZH_MAX=1.0,    # zh_max = RC grid upper edge (Binning.hxx)
    PT2_MAX=1.5,                # pT2_max = RC grid upper edge (Binning.hxx)
)


# ─────────────────────────────────────────────────────────────────────────────
def parse_binspec(spec: str) -> dict:
    """Parse 'Q2=1,2,4,8;xB=0,0.2,0.4' → {Q2: [...], xB: [...]}."""
    out = {}
    if not spec:
        return out
    for part in spec.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        dim, vals = part.split("=", 1)
        edges = [float(x) for x in vals.split(",") if x.strip()]
        if len(edges) >= 2:
            out[dim.strip()] = sorted(edges)
    return out


def phi_modulation(phi_rad, A0, Ac, Acc):
    """N(φ) = A0 · [1 + Ac·cos(φ) + Acc·cos(2φ)]."""
    return A0 * (1.0 + Ac * np.cos(phi_rad) + Acc * np.cos(2.0 * phi_rad))


def fit_phi(phi_centres_deg, counts):
    """
    Fit azimuthal distribution in a 4-D (Q², xB, zh, pT²) bin.
    Returns (A0, Ac, Acc, A0_err, Ac_err, Acc_err, fit_ok).
    fit_ok = 1 if scipy converged, 0 otherwise.
    """
    if not HAS_SCIPY:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    mask = counts > 0
    if mask.sum() < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    phi_rad = np.radians(phi_centres_deg[mask])
    y       = counts[mask].astype(float)
    sigma   = np.sqrt(np.maximum(y, 1.0))

    try:
        popt, pcov = curve_fit(
            phi_modulation, phi_rad, y,
            p0=[float(np.mean(y)), 0.0, 0.0],
            sigma=sigma, absolute_sigma=True, maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
        return (float(popt[0]), float(popt[1]), float(popt[2]),
                float(perr[0]), float(perr[1]), float(perr[2]), 1)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="rc_prep: parquet → centroids + phi-fit CSVs for HAPRAD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("parquet", nargs="+",
                    help="Parquet files (wildcards OK)")
    ap.add_argument("--target", required=True,
                    help="Target label, e.g. LD2, CxC")
    ap.add_argument("--out-dir", default=".",
                    help="Output directory (created if absent)")
    ap.add_argument("--bins",
                    help='Override bin edges, e.g. "Q2=1,2,4,8;xB=0,0.2,0.4,0.6,1.1"')

    # Cut thresholds
    ap.add_argument("--q2-min",  type=float, default=_CUTS["Q2_MIN"])
    ap.add_argument("--w-min",   type=float, default=_CUTS["W_MIN"])
    ap.add_argument("--y-min",   type=float, default=_CUTS["Y_MIN"])
    ap.add_argument("--y-max",   type=float, default=_CUTS["Y_MAX"])
    ap.add_argument("--zh-min",  type=float, default=_CUTS["ZH_MIN"])
    ap.add_argument("--zh-max",  type=float, default=_CUTS["ZH_MAX"])
    ap.add_argument("--pt2-max", type=float, default=_CUTS["PT2_MAX"])

    args = ap.parse_args()

    # ── Input files ───────────────────────────────────────────────────────────
    file_list = sorted(set(f for p in args.parquet for f in glob.glob(p)))
    if not file_list:
        print(f"Error: no files matched: {args.parquet}"); sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Bin edges ─────────────────────────────────────────────────────────────
    user_bins = parse_binspec(args.bins or "")
    bins = {d: user_bins.get(d, DEFAULT_BINS[d])
            for d in ("Q2", "xB", "zh", "pT2", "phi")}

    print("=" * 65)
    print(f"  haprad_rc_input.py  —  target: {args.target}")
    print(f"  Files  : {len(file_list)}")
    print(f"  Out    : {args.out_dir}")
    print("  RC grid bin edges:")
    for d, edges in bins.items():
        print(f"    {d:<5}: {edges}")
    print(f"  Cuts   : Q2>{args.q2_min}, W>{args.w_min}, "
          f"{args.y_min}<y<{args.y_max}, "
          f"{args.zh_min}<zh<{args.zh_max}, pT2<{args.pt2_max}")
    print("=" * 65)

    # ── Histograms ────────────────────────────────────────────────────────────
    dims_5d = ["Q2", "xB", "zh", "pT2", "phi"]
    bh_axes = [bh.axis.Variable(bins[d]) for d in dims_5d]

    h_pip  = bh.Histogram(*bh_axes, storage=bh.storage.Weight())

    # Mean trackers for all 5 analysis dims + nu (both stored per bin)
    track_dims = dims_5d + ["nu"]   # nu is an extra centroid, not a binning axis
    mean_tr = {d: bh.Histogram(*bh_axes, storage=bh.storage.Double())
               for d in track_dims}

    # Consistency tracking: sum of |nu_parquet − nu_computed| / nu_parquet
    consistency_sum = 0.0
    consistency_max = 0.0
    consistency_n   = 0

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i, fpath in enumerate(file_list):
        df = pd.read_parquet(fpath)

        # Pion rows only
        df = df[df["w_pip"] == 1].copy()

        # SIDIS kinematic cuts
        mask = (
            (df["Q2"]  >= args.q2_min)
            & (df["W"]   >= args.w_min)
            & (df["y"]   >= args.y_min)
            & (df["y"]   <= args.y_max)
            & (df["zh"]  >= args.zh_min)
            & (df["zh"]  <= args.zh_max)
            & (df["pT2"] <= args.pt2_max)
        )
        df = df[mask]
        if df.empty:
            continue

        # ── nu ↔ xB consistency check ────────────────────────────────────────
        # nu_computed = Q² / (2 · Mp · xB)  should match parquet's nu column.
        # A large discrepancy means a physics bug in root_2_parquet.py.
        valid_xb = (df["xB"] > 0) & df["nu"].notna() & df["xB"].notna()
        if valid_xb.sum() > 0:
            df_chk = df[valid_xb]
            nu_comp = df_chk["Q2"] / (2.0 * M_PROTON * df_chk["xB"])
            rel_diff = ((nu_comp - df_chk["nu"]).abs() / df_chk["nu"])
            consistency_sum += rel_diff.sum()
            consistency_max  = max(consistency_max, rel_diff.max())
            consistency_n   += len(rel_diff)

        # ── phi_h [0°,360°] → phi_PQ [-180°,180°] ───────────────────────────
        df["phi_PQ"] = np.where(df["phi_h"] > 180.0,
                                df["phi_h"] - 360.0,
                                df["phi_h"])

        # Drop rows with any NaN in required columns
        req = ["Q2", "xB", "zh", "pT2", "phi_PQ", "nu"]
        df = df.dropna(subset=req)
        if df.empty:
            continue

        # ── Fill histograms ──────────────────────────────────────────────────
        vals = [df["Q2"].to_numpy(), df["xB"].to_numpy(),
                df["zh"].to_numpy(), df["pT2"].to_numpy(),
                df["phi_PQ"].to_numpy()]

        h_pip.fill(*vals)

        # Mean tracker: col name → parquet column
        col_map = {"Q2": "Q2", "xB": "xB", "zh": "zh",
                   "pT2": "pT2", "phi": "phi_PQ", "nu": "nu"}
        for d, col in col_map.items():
            mean_tr[d].fill(*vals, weight=df[col].to_numpy())

        if (i + 1) % 5 == 0 or (i + 1) == len(file_list):
            print(f"  Processed {i+1}/{len(file_list)} files …")

    # ── Print consistency check result ────────────────────────────────────────
    print()
    if consistency_n > 0:
        mean_rel = consistency_sum / consistency_n
        print(f"  nu ↔ xB consistency check  ({consistency_n} pions):")
        print(f"    mean |Δnu/nu| = {mean_rel:.2e}   max |Δnu/nu| = {consistency_max:.2e}")
        if mean_rel < 1e-4:
            print("    ✓  nu and xB are consistent in parquet")
        else:
            print("    ✗  WARNING: nu and xB are inconsistent — check root_2_parquet.py")
    else:
        print("  [WARN] No pions passed cuts — cannot do nu↔xB check")

    # ── Build centroids table (5-D) ───────────────────────────────────────────
    print("\nBuilding centroids table …")
    bin_edges = {d: np.array(bins[d]) for d in dims_5d}
    n_bins    = {d: len(bins[d]) - 1  for d in dims_5d}
    phi_cents = 0.5 * (bin_edges["phi"][:-1] + bin_edges["phi"][1:])

    view_n  = h_pip.view().value
    view_mt = {d: mean_tr[d].view() for d in track_dims}

    centroid_rows = []
    for idx in product(*(range(n_bins[d]) for d in dims_5d)):
        n_pip = view_n[idx]
        if n_pip <= 0:
            continue

        row = {}
        for j, d in enumerate(dims_5d):
            lo = bin_edges[d][idx[j]]
            hi = bin_edges[d][idx[j] + 1]
            row[f"{d}_lo"]   = lo
            row[f"{d}_hi"]   = hi
            row[f"{d}_mean"] = view_mt[d][idx] / n_pip   # true centroid

        # nu_mean: from parquet nu column (HAPRAD input)
        # nu_computed: from Q²/(2·Mp·xB) using the centroid values — cross-check
        xb_mean = row["xB_mean"]
        q2_mean = row["Q2_mean"]
        row["nu_mean"]     = view_mt["nu"][idx] / n_pip
        row["nu_computed"] = (q2_mean / (2.0 * M_PROTON * xb_mean)
                              if xb_mean > 0 else np.nan)
        row["nu_xb_agree"] = (
            abs(row["nu_mean"] - row["nu_computed"]) / row["nu_mean"] < 0.05
            if xb_mean > 0 and row["nu_mean"] > 0 else False
        )
        row["N_pip"] = n_pip
        centroid_rows.append(row)

    df_cent = pd.DataFrame(centroid_rows)

    # Column order: bin edges → means (all dims + nu) → N_pip
    col_order = (
        [f"{d}_{s}" for d in dims_5d for s in ("lo", "hi", "mean")]
        + ["nu_mean", "nu_computed", "nu_xb_agree", "N_pip"]
    )
    df_cent = df_cent[[c for c in col_order if c in df_cent.columns]]

    cent_path = os.path.join(args.out_dir, f"centroids_{args.target}.csv")
    df_cent.to_csv(cent_path, index=False)
    print(f"  Wrote {len(df_cent)} bins → {cent_path}")

    # ── Check per-bin nu consistency ─────────────────────────────────────────
    if "nu_xb_agree" in df_cent.columns:
        n_disagree = (~df_cent["nu_xb_agree"]).sum()
        if n_disagree == 0:
            print("  ✓  All bins: nu_mean and nu_computed agree within 5%")
        else:
            print(f"  ✗  {n_disagree} bins: nu_mean vs nu_computed differ > 5%")
            bad = df_cent[~df_cent["nu_xb_agree"]][
                ["Q2_lo","Q2_hi","xB_lo","xB_hi","nu_mean","nu_computed"]
            ]
            print(bad.to_string(index=False))

    # ── Build phi-fit table (4-D: Q², xB, zh, pT²) ───────────────────────────
    print("\nBuilding phi-modulation fit table …")
    dims_4d  = ["Q2", "xB", "zh", "pT2"]
    fit_rows = []

    for idx4 in product(*(range(n_bins[d]) for d in dims_4d)):
        phi_counts = view_n[idx4 + (slice(None),)]   # shape (n_phi,)
        n_total    = phi_counts.sum()
        if n_total <= 0:
            continue

        A0, Ac, Acc, A0e, Ace, Acce, fit_ok = fit_phi(phi_cents, phi_counts)

        row = {}
        for j, d in enumerate(dims_4d):
            row[f"{d}_lo"] = bin_edges[d][idx4[j]]
            row[f"{d}_hi"] = bin_edges[d][idx4[j] + 1]

        row.update({
            "A0": A0, "Ac": Ac, "Acc": Acc,
            "A0_err": A0e, "Ac_err": Ace, "Acc_err": Acce,
            "fit_ok": fit_ok,
            "N_phi_bins_used": int((phi_counts > 0).sum()),
            "N_pip_total": int(n_total),
        })
        fit_rows.append(row)

    df_fits = pd.DataFrame(fit_rows)
    fit_col_order = (
        [f"{d}_{s}" for d in dims_4d for s in ("lo", "hi")]
        + ["A0", "Ac", "Acc", "A0_err", "Ac_err", "Acc_err",
           "fit_ok", "N_phi_bins_used", "N_pip_total"]
    )
    df_fits = df_fits[[c for c in fit_col_order if c in df_fits.columns]]

    fit_path = os.path.join(args.out_dir, f"phi_fits_{args.target}.csv")
    df_fits.to_csv(fit_path, index=False)
    print(f"  Wrote {len(df_fits)} 4-D bins → {fit_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    n_possible_5d = 1
    n_possible_4d = 1
    for d in dims_5d:
        n_possible_5d *= n_bins[d]
    for d in dims_4d:
        n_possible_4d *= n_bins[d]

    print("\n" + "=" * 65)
    print(f"  DONE  —  target: {args.target}")
    print(f"  5-D bins filled : {len(df_cent)} / {n_possible_5d}")
    print(f"  4-D bins filled : {len(df_fits)} / {n_possible_4d}")
    if HAS_SCIPY and len(df_fits) > 0:
        print(f"  Phi fits OK     : {int(df_fits['fit_ok'].sum())} / {len(df_fits)}")
    print(f"  centroids  → {cent_path}")
    print(f"  phi_fits   → {fit_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
