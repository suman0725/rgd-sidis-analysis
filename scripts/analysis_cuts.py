# =============================================================================
# analysis_cuts.py
#
# Single home for all DIS and SIDIS kinematic cut sets and cut functions.
#
# FILE RESPONSIBILITIES IN THIS PROJECT:
#   physics_constants.py  — true physical constants only (masses, E_BEAM, etc.)
#   electron_cuts.py      — electron detector and PID cuts
#   pip_cuts.py           — pion detector and PID cuts
#   common_cuts.py        — detector geometry helpers (is_fd, detect_polarity)
#   analysis_cuts.py      — DIS/SIDIS kinematic cut sets + apply functions (THIS FILE)
#   root_2_parquet.py     — produces parquet files with NO physics cuts applied
#   notebooks (01/02/03)  — import apply_dis_cuts / apply_sidis_cuts here
#
# HOW TO USE:
#   from analysis_cuts import apply_dis_cuts, apply_sidis_cuts
#   df_cut = apply_sidis_cuts(df)
#
# HOW TO SWITCH CUT SETS:
#   Change ONE line only:  CUT_SET = 'standard'
#   Options: 'loose', 'standard', 'tight'
#   The change propagates automatically to all scripts and notebooks.
#
# HOW TO ADD A NEW CUT SET:
#   Add a new entry to CUT_SETS dict with a descriptive name.
#   Set CUT_SET to that name. Done.
# =============================================================================

# =============================================================================
# CUT SET DEFINITIONS
# =============================================================================
# Each cut set is a dictionary of threshold values.
# Add new cut sets here as your analysis evolves.

CUT_SETS = {

    # ── loose ─────────────────────────────────────────────────────────────
    # Wide open cuts for exploratory studies and full kinematic range checks.
    # Recommended for: notebook 01_gen_reco_data_raw (no-cuts sanity check)
    'loose': dict(
        Q2_MIN  = 1.0,   # GeV^2 — photon virtuality lower bound
        W_MIN   = 2.0,   # GeV   — hadronic invariant mass lower bound
        Y_MIN   = 0.10,  # —      inelasticity lower bound
        Y_MAX   = 0.85,  # —      inelasticity upper bound
        ZH_MIN  = 0.1,   # —      hadron energy fraction lower bound
        ZH_MAX  = 0.7,   # —      hadron energy fraction upper bound
        PT2_MAX = 2.0,   # GeV^2 — transverse momentum squared upper bound
    ),

    # ── standard ──────────────────────────────────────────────────────────
    # Matches RG-D preliminary analysis reference cuts.
    # Reference: Q2>1, W>2, 0.25<y<0.85, 0.3<zh<0.7, pT2<1.2 GeV2
    # Recommended for: main physics analysis in notebooks 02 and 03
    'standard': dict(
        Q2_MIN  = 1.0,   # GeV^2 — photon virtuality lower bound
        W_MIN   = 2.0,   # GeV   — hadronic invariant mass lower bound
        Y_MIN   = 0.25,  # —      inelasticity lower bound (removes low-nu edge)
        Y_MAX   = 0.85,  # —      inelasticity upper bound (removes radiative region)
        ZH_MIN  = 0.3,   # —      hadron energy fraction lower bound (removes soft pions)
        ZH_MAX  = 0.7,   # —      hadron energy fraction upper bound (removes exclusive region)
        PT2_MAX = 1.2,   # GeV^2 — transverse momentum squared upper bound
    ),

    # ── tight ─────────────────────────────────────────────────────────────
    # Conservative cuts to reduce systematics at the edges of phase space.
    # Recommended for: systematic uncertainty studies
    'tight': dict(
        Q2_MIN  = 1.5,   # GeV^2 — stricter photon virtuality lower bound
        W_MIN   = 2.0,   # GeV   — hadronic invariant mass lower bound
        Y_MIN   = 0.25,  # —      inelasticity lower bound
        Y_MAX   = 0.80,  # —      stricter inelasticity upper bound
        ZH_MIN  = 0.3,   # —      hadron energy fraction lower bound
        ZH_MAX  = 0.7,   # —      hadron energy fraction upper bound
        PT2_MAX = 1.0,   # GeV^2 — stricter transverse momentum squared upper bound
    ),
}

# =============================================================================
# ACTIVE CUT SET — CHANGE THIS ONE LINE TO SWITCH CUTS EVERYWHERE
# =============================================================================
CUT_SET = 'standard'   # <-- CHANGE to 'loose', 'standard', or 'tight'

# Unpack active cut set into module-level variables.
# Do not change below this line — edit the dict entries above instead.
Q2_MIN  = CUT_SETS[CUT_SET]['Q2_MIN']
W_MIN   = CUT_SETS[CUT_SET]['W_MIN']
Y_MIN   = CUT_SETS[CUT_SET]['Y_MIN']
Y_MAX   = CUT_SETS[CUT_SET]['Y_MAX']
ZH_MIN  = CUT_SETS[CUT_SET]['ZH_MIN']
ZH_MAX  = CUT_SETS[CUT_SET]['ZH_MAX']
PT2_MAX = CUT_SETS[CUT_SET]['PT2_MAX']


# =============================================================================
# CUT FUNCTIONS
# =============================================================================
# Import these in any script or notebook.
# Thresholds come from the active CUT_SET above — no need to pass values.

def apply_dis_cuts(df):
    """
    Apply standard DIS kinematic cuts to a DataFrame.
    Cuts applied: Q2 > Q2_MIN, W > W_MIN, Y_MIN < y < Y_MAX
    Thresholds are set by CUT_SET in analysis_cuts.py.
    Returns a filtered copy of df.
    """
    mask = (
        (df['Q2'] > Q2_MIN) &
        (df['W']  > W_MIN)  &
        (df['y']  > Y_MIN)  &
        (df['y']  < Y_MAX)
    )
    return df[mask].copy()


def apply_sidis_cuts(df):
    """
    Apply DIS + SIDIS kinematic cuts to a DataFrame.
    Cuts applied: DIS cuts + ZH_MIN < zh < ZH_MAX + pT2 < PT2_MAX
    Thresholds are set by CUT_SET in analysis_cuts.py.
    Returns a filtered copy of df.
    """
    df = apply_dis_cuts(df)
    mask = (
        (df['zh']  > ZH_MIN)  &
        (df['zh']  < ZH_MAX)  &
        (df['pT2'] < PT2_MAX)
    )
    return df[mask].copy()


def get_active_cuts():
    """
    Return a summary string of the currently active cut set.
    Useful for printing in notebooks to document which cuts were applied.
    """
    return (
        f"Active cut set: '{CUT_SET}'\n"
        f"  Q2  > {Q2_MIN} GeV^2\n"
        f"  W   > {W_MIN} GeV\n"
        f"  {Y_MIN} < y < {Y_MAX}\n"
        f"  {ZH_MIN} < zh < {ZH_MAX}\n"
        f"  pT2 < {PT2_MAX} GeV^2"
    )
