#!/usr/bin/env python3
"""
plot_phi_fits.py
EG2-style phi panels: fit N(phi)=A+B*cos(phi)+C*cos(2phi) in fixed (Q2,xB,zh) bins.
Usage kept backward-compatible:
  plot_phi_fits.py TARGET PARQUET_GLOB FITCSV OUTPNG
"""
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = sys.argv[1]
PARQUET = sys.argv[2]
OUTPNG = sys.argv[4]

# EG2-like plot binning for visualization
Q2_PLOT = [1.0, 2.5, 8.0]          # 2 bins
XB_PLOT = [0.1, 0.3, 0.75]         # 2 bins
ZH_PLOT = [0.3, 0.45, 0.6, 0.75]   # 3 bins
PHI_EDGES = np.linspace(0.0, 180.0, 25)  # 24 bins
PHI_CENTS = 0.5 * (PHI_EDGES[:-1] + PHI_EDGES[1:])
PHI_FINE = np.linspace(0.0, 180.0, 400)


def fit_abc(phi_deg, y):
    phi = np.radians(phi_deg)
    xmat = np.column_stack([np.ones_like(phi), np.cos(phi), np.cos(2.0 * phi)])
    w = 1.0 / np.sqrt(np.maximum(y, 1.0))
    xw = xmat * w[:, None]
    yw = y * w
    beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    a, b, c = beta
    return a, b, c


files = sorted(glob.glob(PARQUET))
if not files:
    raise SystemExit(f"No parquet files matched: {PARQUET}")

print(f"Loading {len(files)} parquet files …")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

df = df[df["w_pip"] == 1].copy()
df = df[
    (df["Q2"] >= 1.0) & (df["W"] >= 2.0) & (df["y"] >= 0.25) & (df["y"] <= 0.85)
    & (df["zh"] >= 0.3) & (df["zh"] <= 0.75) & (df["pT2"] <= 1.5)
]
df["phi_PQ"] = np.where(df["phi_h"] > 180.0, df["phi_h"] - 360.0, df["phi_h"])
df["phi_abs"] = np.abs(df["phi_PQ"])
df = df.dropna(subset=["Q2", "xB", "zh", "phi_abs"])
print(f"  {len(df)} pions after cuts")

nrows = (len(Q2_PLOT) - 1) * (len(XB_PLOT) - 1)
ncols = len(ZH_PLOT) - 1
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.3, nrows * 2.9), sharex=True)
axes = np.array(axes).reshape(nrows, ncols)

panel = 0
for iq2 in range(len(Q2_PLOT) - 1):
    q2lo, q2hi = Q2_PLOT[iq2], Q2_PLOT[iq2 + 1]
    for ixb in range(len(XB_PLOT) - 1):
        xblo, xbhi = XB_PLOT[ixb], XB_PLOT[ixb + 1]
        row = panel
        panel += 1

        for izh in range(len(ZH_PLOT) - 1):
            zhlo, zhhi = ZH_PLOT[izh], ZH_PLOT[izh + 1]
            ax = axes[row, izh]

            sel = df[
                (df["Q2"] >= q2lo) & (df["Q2"] < q2hi)
                & (df["xB"] >= xblo) & (df["xB"] < xbhi)
                & (df["zh"] >= zhlo) & (df["zh"] < zhhi)
            ]["phi_abs"].to_numpy()

            counts, _ = np.histogram(sel, bins=PHI_EDGES)
            yerr = np.sqrt(np.maximum(counts, 1))
            ax.errorbar(PHI_CENTS, counts, yerr=yerr, fmt="o", color="black", ms=3, lw=1, label="data")

            if counts.sum() >= 30 and np.count_nonzero(counts) >= 8:
                a, b, c = fit_abc(PHI_CENTS, counts)
                fit = a + b * np.cos(np.radians(PHI_FINE)) + c * np.cos(2.0 * np.radians(PHI_FINE))
                ax.plot(PHI_FINE, fit, color="red", lw=1.6, label="fit")
                ac = b / a if abs(a) > 1e-12 else np.nan
                acc = c / a if abs(a) > 1e-12 else np.nan
                ax.text(0.03, 0.93, f"Ac={ac:.2f}, Acc={acc:.2f}", transform=ax.transAxes, fontsize=7, va="top")

            ax.set_title(f"z[{zhlo:.2f},{zhhi:.2f}]  Q²[{q2lo:.1f},{q2hi:.1f}]  xB[{xblo:.2f},{xbhi:.2f}]",
                         fontsize=7)
            ax.set_xlim(0, 180)
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.tick_params(labelsize=7)
            if izh == 0:
                ax.set_ylabel("Counts", fontsize=8)
            if row == nrows - 1:
                ax.set_xlabel(r"$\phi$ [deg]", fontsize=8)

handles, labels = axes[0, 0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="upper right", fontsize=8)
fig.suptitle(f"{TARGET}: EG2-style phi slices (raw points + A+Bcos(phi)+Ccos(2phi) fit)", fontsize=11, y=0.995)
plt.tight_layout()
plt.savefig(OUTPNG, dpi=140, bbox_inches="tight")
print(f"Saved → {OUTPNG}")
