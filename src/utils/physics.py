
# Imports
import numpy as np

# ------------------------ Constants ----------------------- #
ATOMIC_MASS = 1.66053906660e-27 # kg
NUCLEON_MASS = 1.672e-27 # kg
ELECTRON_MASS = 9.11e-31
BOLTZMAN_CONSTANT = 1.38064e-23 # J K−1
ELECTRON_CHARGE = -1.6e-19 # C
ELECTRON_EFFECTIVE_DIAMETER = 2.8179403227e-15 # m

# ------------------------ Mass ---------------------------- #
def get_mass_part(electrons_nb, protons_number, neutrons_number):
    return (neutrons_number+protons_number)*NUCLEON_MASS+electrons_nb*ELECTRON_MASS

# ------------------------ gaussian distribution ----------------- #
def gaussian(temperature, mass):
    v_mean = np.sqrt(BOLTZMAN_CONSTANT*temperature/mass)
    return v_mean

# ------------------------ Maxwellian distribution ----------------- #
# mean speed
def maxwellian_mean_speed(T,m):
    return np.sqrt(8.0*BOLTZMAN_CONSTANT*T/(np.pi*m))

# Maxwellian flux
def maxwellian_flux(density, v_mean, drift = 0):
    return 0.25*density*v_mean+density*drift

# ---------------------- Mean free path - time ------------------------- #

def mean_free_path(cross_section, density):
    return 1/(cross_section*density)

def mean_free_time(mfp, v_mean):
    return mfp/v_mean
    