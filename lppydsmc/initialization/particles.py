from ..utils.estimation import is_inside_polygon # useful to initialize point on a given system and make sure we are inside
from ..utils.physics import gaussian

import numpy as np

def initialize(points:np.ndarray, quantity:int, t:str, params:list):
    velocities = dispatcher(t)(quantity, *params)
    positions = uniform_position(quantity, points)
    return np.concatenate((positions, velocities), axis = 1)
    
def dispatcher(value): # simple dispatcher for now as we did not add any other velocity init
    return maxwelllian

def maxwelllian(quantity, temperature, mass):
    vel_std = gaussian(temperature, mass)
    vx = np.random.normal(loc=0.0, scale=vel_std, size = quantity)
    vy = np.random.normal(loc=0.0, scale=vel_std, size = quantity)
    vz = np.random.normal(loc=0.0, scale=vel_std, size = quantity)
    return np.stack((vx,vy,vz), axis = 1) 

def uniform_position(quantity, points, warnings = True):
    min_x, max_x = np.min(points[:,0]), np.max(points[:,0])
    min_y, max_y = np.min(points[:,1]), np.max(points[:,1])
    
    # taking the smaller rectangle in which the polygone is
    rectangle_area  = (max_x-min_x)*(max_y-min_y) 

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
