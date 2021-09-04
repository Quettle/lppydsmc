from ..utils.estimation import is_inside_polygon # useful to initialize point on a given system and make sure we are inside
from ..utils.physics import gaussian

import numpy as np

def initialize(points:np.ndarray, quantity:int, t:str, params:list):
    """ Returns a array of new particles described by [x,y,vx,vy,vz] with the position (x,y) being uniformly 
    initialized in the system defined by the points and the velocity (vx,vy,vz) being initialized as specified by *t*.

    Args:
        points (np.ndarray): the points describing the system. Should be given in counter-clock wise order.
        quantity (int): the quantity to initialized.
        t (str): the type of the velocity initialization. Available : 'maxwellian', 'uniform'
        params (list): parameters used for velocity initialization. Contains :
                            - 'maxwellian' : [temperature, mass]
                            - 'uniform' : [min velocity, max velocity] ; dimensions is supposed for now to be three.

    Returns:
        np.ndarray: 2D-arrays of shape (quantity x 5) describing the initialized particles
    """
    velocities = dispatcher(t)(quantity, *params)
    positions = uniform_position(quantity, points)
    return np.concatenate((positions, velocities), axis = 1)
    
def dispatcher(value):
    """ Simple dispatcher: associated the right velocity initialization function depending on value.

    Args:
        value (str): the type choosen as initialization.
    """
    if(value == 'maxwellian'):
        return maxwellian
    elif(value == 'uniform'):
        return uniform_speed
    else :
        print(f'Value {value} not recognized. Returning maxwellian by default.')
        return maxwellian

def maxwellian(quantity, temperature, mass):
    """ Performs a maxwellian velocity initialization.

    Args:
        quantity (int): quantity of particles to initialize
        temperature (float): temperature of the maxwellian
        mass (float): mass of the species

    Returns:
        np.ndarray: a 2D-array of velocites, of shape (quantity, 3), a velocity is describe as [vx,vy,vz]
    """
    vel_std = gaussian(temperature, mass)
    vx = np.random.normal(loc=0.0, scale=vel_std, size = quantity)
    vy = np.random.normal(loc=0.0, scale=vel_std, size = quantity)
    vz = np.random.normal(loc=0.0, scale=vel_std, size = quantity)
    return np.stack((vx,vy,vz), axis = 1) 

def uniform_speed(quantity, min_vel, max_vel, dimensions = 3):
    """ Performs a uniform velocity initialization.

    Args:
        quantity (int): quantity of particles to initialize
        min_vel (float): minimum velocity
        max_vel (float): maximum velocity
        dimensions (int, optional) : the dimensions of the velocity. Default to 3.

    TODO : add the compatibility with dimensions. The idea would be to update only one dimension along the 3 for example.

    Returns:
        np.ndarray: a 2D-array of velocites, of shape (quantity, 3), a velocity is describe as [vx,vy,vz]
    """
    velocities = np.zeros((quantity, dimensions))
    for k in range(dimensions):
        velocities[:,k] = np.random.uniform(low = min_vel, high = max_vel, size = (quantity))
    return velocities

def uniform_position(quantity, points, warnings = True):
    """ Initiaized positions on the segments defined by points given in counter-clock order.

    Args:
        quantity (int): number of positions to initialize
        points (np.ndarray): 2D-arrays describing the system, a point is describe by (x,y).
        warnings (bool, optional): Limit of tries before terminating early the algorithm. Defaults to True.

    Returns:
        np.ndarray: 2D-array of shape (quantity x 2), a position is described by (x,y)
    """
    min_x, max_x = np.min(points[:,0]), np.max(points[:,0])
    min_y, max_y = np.min(points[:,1]), np.max(points[:,1])
    
    count = 0
    positions = np.zeros((quantity,2))

    if(warnings): tries = 0
    while(count < quantity):
        if(tries):
            tries+=1
            if(tries > 10):
                print('WARNING : Reached number of tries limit. Initialized particles {}/{}.'.format(count,quantity))
                return positions

        # sampling points
        X_samples = min_x + np.random.random(size = (quantity))*(max_x-min_x)
        Y_samples = min_y + np.random.random(size = (quantity))*(max_y-min_y)  
        sample_points = np.stack((X_samples, Y_samples), axis = 1)
        for point in sample_points: 
            b = is_inside_polygon(list(points),tuple(point))
            if(b):
                positions[count,:] = point
                count += 1
                
                if(count == quantity): return positions
