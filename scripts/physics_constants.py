import numpy as np

# Energy (GeV) 
E_BEAM = 10.54 

# Masses (GeV)
M_PROTON   = 0.9382720813
M_NEUTRON  = 0.9395654133
M_NUCLEON  = M_PROTON       # default target mass for xB, W
M_ELECTRON = 0.0005109989

# CHANGE THIS: Rename or add M_PION_PLUS to match bank_builders.py
M_PION_PLUS = 0.13957018 

# ADD THIS: Speed of light in cm/ns (required by bank_builders.py)
C_VEL = 29.9792458 

# Angle conversions
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# NOTE: DIS/SIDIS kinematic cut sets and cut functions have moved to:
#       analysis_cuts.py  — edit that file to change cut values or add new cut sets.