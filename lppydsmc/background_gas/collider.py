import numpy as np
from numpy.linalg.linalg import norm

# NOTE: For now, the gas is supposed to be monoatomic.
# NOTE: Schullian (2019) proposes something that could be good doing, however the version used here is much simpler
# In the future, the implementation of a better model is most likely required. Check with the aforementionned article.

def handler_particles_collisions(arr, dt, radius, mass, gas_radius, gas_mass, gas_density_fn, gas_dynamic_fn = None, monitoring = False): # INPLACE
    # select couples that ought to collide
    # the frequency of collision is usually given by : nu = cr/mean_free_path

    cross_section = np.pi*(radius+gas_radius)**2
    local_densities = gas_density_fn(arr[:,0],arr[:,1]) # giving the positions of each particle to the gas
    # gas_density_fn returns an array with the density of the background gas for each particle of the arr

    if(gas_dynamic_fn is not None):
        local_dynamics = gas_dynamic_fn(arr[:,0],arr[:,1]) # returns a vector of size arr.shape[0]x3 
                # this vector contains the velocity of the other colliding particle if they are to happen 
        vr_norm = np.linalg.norm((arr[:,2:]-local_dynamics), axis = 1)
    else:
        vr_norm = np.linalg.norm(arr[:,2:], axis = 1)
    proba = local_densities*cross_section*vr_norm*dt  # the collision fréquency for a particle times the time step, this is the expected number of collision in the time step
    # it is important to be careful to have a dt small enough so this proba is never bigger than one...

    if(proba.shape[0]>0 and np.max(proba)>1): print('In collision with background gas, a collision with probability higher than one was encountered. You should lower *dt*.')
    
    r = np.random.random(size = proba.shape)
    colliding = np.where(proba>r)[0]
    arr_colliding = arr[colliding]

    local_dynamics_colliding = np.zeros(arr_colliding.shape)

    if(gas_dynamic_fn is not None):
        local_dynamics_colliding[:,2:] = local_dynamics[colliding]
        
    arr[colliding] =  reflect(np.stack((arr_colliding, local_dynamics_colliding), axis = 1), vr_norm[colliding], mass, gas_mass)[:,0,:]

    # if(monitoring):
    #     return colliding

    return colliding # returning by default the indexes of the colliding particles

# almost same function as in advection.collider

def reflect(arr, vr_norm, mass1, mass2):
    """ Reflect particles following their collisions. Reflections are for now purely specular (no randomness).

    Args:
        arr (np.ndarray): 3D-array of size (number of actually colliding couples x 2 x 5)
        vr_norm (np.ndarray): relative velocity norm for each colliding couple
        masses (np.ndarray, optional): Mass for each particle, for each couple. If mass is the same for every particle, then use None as it simplified the computations.
                                       Defaults to None.

    Returns:
        np.ndarray  : Returns the new array with velocities having been updated accordingly.
                      Note that *arr* is in fact changed in place, however, we still return it in case it is needed  (in the case of 
                      the main algorithm *handler_particles_collisions*, *arr* is a copy and not a reference to the initial array).
                      E.g. arr = array[is_colliding(proba)]
    """
    m = mass1+mass2
    coeff1 = mass1/m # float
    coeff2 = mass2/m

    r = np.random.random(size = (2,arr.shape[0]))
    ctheta = 2*r[0,:]-1
    stheta = np.sqrt(1-ctheta*ctheta)
    phi = 2*np.pi*r[1,:]
    
    v_cm = coeff1*arr[:,0,2:]+coeff2*arr[:,1,2:]
    v_r_ = np.expand_dims(vr_norm, axis = 1)*np.stack((stheta*np.cos(phi), stheta*np.sin(phi), ctheta),  axis = 1) # 

    arr[:,0,2:] = v_cm + coeff1*v_r_
    arr[:,1,2:] = v_cm - coeff2*v_r_

    return arr