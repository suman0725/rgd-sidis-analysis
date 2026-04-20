# =============================================================================
# config.py  —  ONE file to change when moving to a new machine
# =============================================================================
# Edit the paths in this block only. Everything else auto-derives from them.
# =============================================================================

import os

# ── Root directories ──────────────────────────────────────────────────────────

# Directory containing the parquet data files (sidis_*, gen_*, reco_*)
PARQUET_DIR = '/Users/sumanshrestha/Desktop/Physics_Analysis/test_output'

# Directory containing yields CSVs produced by count_sidis_bins.py
YIELDS_DIR  = os.path.join(PARQUET_DIR, 'yields')

# Root of this repo (Final_Code folder)
REPO_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where notebooks save output PDFs / result CSVs
OUTPUT_DIR  = REPO_DIR

# ── Physics settings ──────────────────────────────────────────────────────────

P_BEAM = 0.85   # beam polarization — update with exact RGD run-averaged value

# ── Target / polarity defaults ────────────────────────────────────────────────

TARGET   = 'LD2'
POLARITY = 'OB'

# ── Derived paths (do not edit) ───────────────────────────────────────────────

SCRIPTS_DIR = os.path.join(REPO_DIR, 'scripts')

def data_file(target=TARGET, polarity=POLARITY, suffix='testnewLD2.root.parquet'):
    """Return path to a SIDIS data parquet file."""
    return os.path.join(PARQUET_DIR, f'sidis_{target}_{polarity}_{suffix}')
