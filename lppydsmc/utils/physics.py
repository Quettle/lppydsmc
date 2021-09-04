# Imports
import numpy as np

"""
This module defined useful functions and constants.

List of constants (dimension):
    ATOMIC_MASS (kg)
    NUCLEON_MASS (kg)
    ELECTRON_MASS (kg)
    BOLTZMAN_CONSTANT (J.K−1)
    ELECTRON_CHARGE (C)
    ELECTRON_EFFECTIVE_DIAMETER (m)
    AVOGADRO_CONSTANT (mol-1)
    GAS_CONSTANT (J.mol-1.K-1)
    VACUUM_PERMITTIVITY (F.m-1)
    PA_TO_TORR (Torr/Pa)
"""

# ------------------------ Constants ----------------------- #
ATOMIC_MASS = 1.66053906660e-27 # kg
NUCLEON_MASS = 1.672e-27 # kg
ELECTRON_MASS = 9.11e-31
BOLTZMAN_CONSTANT = 1.38064e-23 # J K−1
ELECTRON_CHARGE = -1.6e-19 # C
ELECTRON_EFFECTIVE_DIAMETER = 2.8179403227e-15 # m
AVOGADRO_CONSTANT = 6.02e23 # mol-1
GAS_CONSTANT = 8.314 # J.mol-1.K-1
VACUUM_PERMITTIVITY = 8.8541878128e-12 # F.m-1

# ------------------------ Conversion ------------------- #
J_TO_EV = 1/1.6e-19
PA_TO_TORR = 1/(101325/760.) # Torr/Pa - 1 torr is 1/760 of a standard atmosphere (= 101325 Pa)

# ------------------------ Mass ---------------------------- #
def get_mass_part(electrons_nb, protons_number, neutrons_number):
    """ Return the mass of a particle based on the number of electrons / protons / neutrons.
    Note that is functions support both integers, floats and arrays of integers / floats.

    Args:
        electrons_nb (int, np.ndarray): the number of electrons
        protons_number (int, np.ndarray): the number of protons
        neutrons_number (int, np.ndarray): the number of neutrons

    Returns:
        int, np.ndarray: the mass of the particle(s)
    """
    return (neutrons_number+protons_number)*NUCLEON_MASS+electrons_nb*ELECTRON_MASS

# ------------------------ gaussian distribution ----------------- #
def gaussian(temperature, mass):
    std = np.sqrt(BOLTZMAN_CONSTANT*temperature/mass)
    return std

# ------------------------ Maxwellian distribution ----------------- #
# mean speed
def maxwellian_mean_speed(T,m):
    return np.sqrt(8.0*BOLTZMAN_CONSTANT*T/(np.pi*m))

# Maxwellian flux
def maxwellian_flux(density, v_mean, drift = 0):
    return 0.25*density*v_mean+density*drift

# ---------------------- Mean free path - time ------------------------- #

def mean_free_path(cross_section, density): 
    """ Compute the mean free path for a monoatomic gas. Note : this function supports vectorization.

    Args:
        cross_section (float): the cross section of the species
        density (float): the density in the system

    Returns:
        float: return the mean free path for a monoatomic gas at gas density *density*.
    """
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