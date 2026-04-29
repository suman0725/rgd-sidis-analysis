#!/usr/bin/env python3
"""
plot_phi_fits.py
Plot phi distributions + fitted curves for each 4D (Q², xB, zh, pT²) bin.
Reads parquet directly so we see real data points + fit overlaid.
"""
import glob, sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET  = sys.argv[1]   # LD2 or CxC
PARQUET = sys.argv[2]   # glob pattern e.g. "/path/*.parquet"
FITCSV  = sys.argv[3]   # phi_fits_{TARGET}.csv
OUTPNG  = sys.argv[4]   # output PNG path

M_PROTON = 0.9382720813

# ── RC grid (must match haprad_rc_input.py) ──────────────────────────────────
Q2_EDGES  = [1.0, 2.0, 4.0, 8.0]
XB_EDGES  = [0.1, 0.2, 0.3, 0.4, 0.6, 0.75]
ZH_EDGES  = [0.3, 0.5, 0.75, 1.0]
PT2_EDGES = [0.0, 0.5, 1.0, 1.5]
PHI_EDGES = [-180, -108, -36, 36, 108, 180]
PHI_CENTS = [0.5*(PHI_EDGES[i]+PHI_EDGES[i+1]) for i in range(5)]

# ── Load parquet ──────────────────────────────────────────────────────────────
files = sorted(glob.glob(PARQUET))
print(f"Loading {len(files)} parquet files …")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# Pion rows + SIDIS cuts
df = df[df["w_pip"] == 1].copy()
df = df[(df["Q2"]>=1.0)&(df["W"]>=2.0)&(df["y"]>=0.25)&(df["y"]<=0.85)
        &(df["zh"]>=0.3)&(df["zh"]<=1.0)&(df["pT2"]<=1.5)]
df["phi_PQ"] = np.where(df["phi_h"]>180, df["phi_h"]-360, df["phi_h"])
df = df.dropna(subset=["Q2","xB","zh","pT2","phi_PQ"])
print(f"  {len(df)} pions after cuts")

# ── Assign 4D bin labels ──────────────────────────────────────────────────────
df["iQ2"]  = pd.cut(df["Q2"],  bins=Q2_EDGES,  labels=False, right=True)
df["ixB"]  = pd.cut(df["xB"],  bins=XB_EDGES,  labels=False, right=True)
df["izh"]  = pd.cut(df["zh"],  bins=ZH_EDGES,  labels=False, right=True)
df["ipT2"] = pd.cut(df["pT2"], bins=PT2_EDGES, labels=False, right=True)
df["iphi"] = pd.cut(df["phi_PQ"], bins=PHI_EDGES, labels=False, right=True)
df = df.dropna(subset=["iQ2","ixB","izh","ipT2","iphi"])
for c in ["iQ2","ixB","izh","ipT2","iphi"]:
    df[c] = df[c].astype(int)

# ── Load fit parameters ───────────────────────────────────────────────────────
fits = pd.read_csv(FITCSV)

# ── Get unique 4D bins sorted by N_pip_total ─────────────────────────────────
bins_4d = (df.groupby(["iQ2","ixB","izh","ipT2"])
             .size().reset_index(name="N")
             .sort_values("N", ascending=False))

# Take top 20 bins (or all if fewer)
top_bins = bins_4d.head(20)
n_plots  = len(top_bins)

# ── Plot ──────────────────────────────────────────────────────────────────────
ncols = 4
nrows = int(np.ceil(n_plots / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.2))
axes = axes.flatten()

phi_fine = np.linspace(-180, 180, 300)

for idx, (_, row) in enumerate(top_bins.iterrows()):
    ax = axes[idx]
    iQ2, ixB, izh, ipT2 = int(row.iQ2), int(row.ixB), int(row.izh), int(row.ipT2)

    # Data: counts per phi bin
    mask = ((df.iQ2==iQ2)&(df.ixB==ixB)&(df.izh==izh)&(df.ipT2==ipT2))
    counts = df[mask].groupby("iphi").size().reindex(range(5), fill_value=0).values

    # Fit params from CSV
    q2lo, q2hi = Q2_EDGES[iQ2], Q2_EDGES[iQ2+1]
    xblo, xbhi = XB_EDGES[ixB], XB_EDGES[ixB+1]
    zhlo, zhhi = ZH_EDGES[izh], ZH_EDGES[izh+1]
    ptlo, pthi = PT2_EDGES[ipT2], PT2_EDGES[ipT2+1]

    frow = fits[(fits.Q2_lo==q2lo)&(fits.Q2_hi==q2hi)
               &(fits.xB_lo==xblo)&(fits.xB_hi==xbhi)
               &(fits.zh_lo==zhlo)&(fits.zh_hi==zhhi)
               &(fits.pT2_lo==ptlo)&(fits.pT2_hi==pthi)]

    ax.bar(PHI_CENTS, counts, width=72, color="steelblue",
           alpha=0.6, label="data", zorder=2)
    ax.errorbar(PHI_CENTS, counts, yerr=np.sqrt(np.maximum(counts,1)),
                fmt='none', color='navy', capsize=3, zorder=3)

    if len(frow) == 1 and frow.iloc[0].fit_ok == 1:
        A0  = frow.iloc[0].A0
        Ac  = frow.iloc[0].Ac
        Acc = frow.iloc[0].Acc
        phi_r = np.radians(phi_fine)
        fit_curve = A0 * (1 + Ac*np.cos(phi_r) + Acc*np.cos(2*phi_r))
        # Scale to bin width (72 deg / 360 deg * total)
        fit_curve *= (72.0/360.0) * counts.sum() / (A0 + 1e-10) * A0 / counts.sum() * counts.sum()
        # Simpler: normalize so integral matches total counts
        norm = counts.sum() / (np.trapezoid(A0*(1+Ac*np.cos(np.radians(phi_fine))+Acc*np.cos(2*np.radians(phi_fine))), phi_fine) / 360 * 5)
        fit_curve = norm * A0 * (1 + Ac*np.cos(phi_r) + Acc*np.cos(2*phi_r))
        color = "red" if abs(Ac)>2 or abs(Acc)>1.5 else "darkorange"
        ax.plot(phi_fine, fit_curve, color=color, lw=1.8,
                label=f"Ac={Ac:.2f}\nAcc={Acc:.2f}")

    ax.set_title(f"Q²[{q2lo:.0f},{q2hi:.0f}] xB[{xblo:.2f},{xbhi:.2f}]\n"
                 f"zh[{zhlo:.2f},{zhhi:.2f}] pT²[{ptlo:.1f},{pthi:.1f}]",
                 fontsize=7)
    ax.set_xlabel("φ [deg]", fontsize=7)
    ax.set_ylabel("N_pip", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180,-108,-36,36,108,180])

# Hide unused panels
for idx in range(n_plots, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(f"φ distributions + fits — {TARGET}  (top {n_plots} bins by N_pip)",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(OUTPNG, dpi=130, bbox_inches="tight")
print(f"Saved → {OUTPNG}")
