# rgd_cuts/pids.py
from enum import IntEnum

class PID(IntEnum):
    ELECTRON   = 11
    POSITRON   = -11
    PROTON     = 2212
    NEUTRON    = 2112
    PION_PLUS  = 211
    PION_MINUS = -211
    KAON_PLUS  = 321
    KAON_MINUS = -321

