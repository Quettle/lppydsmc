import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import lppydsmc as ld

# useful functions

def temperature_evolution(data, mass):
    from lppydsmc.utils.physics import BOLTZMAN_CONSTANT
    # speed norm
    data['v2'] = data['vx']*data['vx']+data['vy']*data['vy']+data['vz']*data['vz']
    data['v'] = np.sqrt(data['v2'])

    # drift 
    v_mean = data.groupby(data.index).mean()
    v_mean['drift2'] = v_mean['vx']*v_mean['vx']+v_mean['vy']*v_mean['vy']+v_mean['vz']*v_mean['vz']
    v_mean['drift'] = np.sqrt(v_mean['drift2'])

    # 3/2 k T = 1/2 m (<v²>-|<v>|²)
    temperature = mass/(3.*BOLTZMAN_CONSTANT)*(v_mean['v2']-v_mean['drift2'])
    # temperature = mass/(3.*BOLTZMAN_CONSTANT)*((v_mean['v']-v_mean['drift'])**2)

    return temperature

def variance_speed_evolution(data):
    data['v2'] = data['vx']*data['vx']+data['vy']*data['vy']+data['vz']*data['vz']
    data['v'] = np.sqrt(data['v2'])
    data_var = data.groupby(data.index).var()
    return data_var['v']

def translational_energy(data, mass):
    data['tx'] = 0.5*mass*data['vx']*data['vx']
    data['ty'] = 0.5*mass*data['vy']*data['vy']
    data['tz'] = 0.5*mass*data['vz']*data['vz']
    
    groups = data[['tx','ty','tz']].groupby(data.index).mean()
    
    return group

def validate_maxwellian_distribution(ax, data, steady_frame, mass, temperature):
    data['tx'] = 0.5*mass*data['vx']*data['vx']
    data['ty'] = 0.5*mass*data['vy']*data['vy']
    data['tz'] = 0.5*mass*data['vz']*data['vz']
    data['t'] =  data['tx']+data['ty']+data['tz']
    
    
    tf = data['t'].loc[data.index>steady_frame]
    
    factor = ld.utils.physics.BOLTZMAN_CONSTANT*temperature
    f = np.exp(-tf/factor)
    
    pre_factor = 1  # target_density # target_density*(mass/(factor*np.pi*2))**(3/2)
    ax.set_yscale('log')
    ax.scatter(tf*ld.utils.physics.J_TO_EV, pre_factor*f, s = 0.1)
    ax.set_xlabel('Translational energy (eV)')
    ax.set_ylabel(r'Probability density $f(E_t, T)$')
    
    return ax