
# Imports
import numpy as np

# ------------------------ Constants ----------------------- #
ATOMIC_MASS = 1.66053906660e-27 # kg
NUCLEON_MASS = 1.672e-27 # kg
ELECTRON_MASS = 9.11e-31
BOLTZMAN_CONSTANT = 1.38064e-23 # J K−1
ELECTRON_CHARGE = -1.6e-19 # C
ELECTRON_EFFECTIVE_DIAMETER = 2.8179403227e-15 # m
AVOGADRO_CONSTANT = 6.02e23 # mol-1
GAS_CONSTANT = 8.314 # J.mol-1.K-1

# ------------------------ Conversion ------------------- #

PA_TO_TORR = 1/(101325/760.) # Torr/Pa - 1 torr is 1/760 of a standard atmosphere (= 101325 Pa)

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

def mean_free_path(cross_section, density): # between two particles of same size
    return 1/(np.sqrt(2)*cross_section*density)

def mean_free_time(mfp, v_mean):
    return mfp/v_mean

def mean_free_path_simu(nb_particles, nb_collisions, time_step, mean_speed):
    return 0.5*nb_particles*time_step*mean_speed/(nb_collisions) # 0.5 because : collisions are between 2 same species particles - we are counting twice too many collisions.
    
# ---------------------- Mass flow rate ------------------------------- #

def compute_mass_flow_rate(qty, delta_time, mass):
    return qty*mass/delta_time

# --------------------- Pressure ------------------------ #

def pressure(nb_parts, volume, temperature): # in Pascal
    return nb_parts*GAS_CONSTANT*temperature/(AVOGADRO_CONSTANT*volume)

def pressure_torr(nb_parts, volume, temperature): # in Pascal
    return pressure(nb_parts, volume, temperature)*PA_TO_TORR

# ---------------------- Speed of sound ------------------------------- #

def speed_of_sound(molecular_mass, temperature, gamma):
    # all units in SI
    # sqrt(gamma R T / M)
    return np.sqrt(8.314*gamma*temperature/molecular_mass)