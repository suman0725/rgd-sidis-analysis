# =============================================================================
# common_cuts.py
#
# Detector geometry helpers only.
# Responsibilities: polarity detection, forward detector (FD) identification.
#
# FILE RESPONSIBILITIES IN THIS PROJECT:
#   physics_constants.py  — true physical constants (masses, E_BEAM, etc.)
#   electron_cuts.py      — electron detector and PID cuts
#   pip_cuts.py           — pion detector and PID cuts
#   common_cuts.py        — detector geometry helpers (is_fd, detect_polarity)  ← THIS FILE
#   analysis_cuts.py      — DIS/SIDIS kinematic cut sets + apply functions
#   root_2_parquet.py     — produces parquet files with NO physics cuts applied
#   notebooks (01/02/03)  — import apply_dis_cuts/apply_sidis_cuts from analysis_cuts.py
# =============================================================================

import uproot
import numpy as np


def detect_polarity(path) -> str:
    """Read RUN_config_torus from the 'data' tree."""
    try:
        with uproot.open(path) as f:
            # Uproot handles 'data;24', 'data;8' etc. automatically
            tree = f["data"]
            torus = tree.arrays("RUN_config_torus", entry_stop=1, library="np")["RUN_config_torus"][0]
            return "OB" if torus > 0 else "IB"
    except Exception as e:
        raise RuntimeError(f"CRITICAL: Could not read 'data' tree or torus from {path}: {e}")


def is_fd(status):
    abs_status = np.abs(np.asarray(status))
    return (abs_status // 1000).astype(int) & 2 > 0

def is_fd_ele(pid, status):
    # Must be PID 11, negative status (trigger), and in FD
    return (pid == 11) & (status < 0) & is_fd(status)

def is_fd_pip(pid, status):
    # Must be PID 211 and in the FD geometry
    # (Pions usually don't need the negative sign)
    return (pid == 211) & is_fd(status)
#is_forward = 2000 <= abs(status) < 4000
