import numpy as np
from random import random
import numpy.ma as ma
import numexpr

from ..utils.physics import gaussian

# ------------------------- reflection default functions ------------------ #

""" The following functions describe reflections on boundaries for colliding particles. These are default functions. 
It is possible to add your own, either directly in the code, throught the configuration file as a string or finally,
if it is defined somewhere else, by passing it to the function *reflect_back_in*. It can be useful in case of use 
in *jupyter notebooks*.
"""

def reflection_functions_dispatcher(value):
    """ Simple dispatcher, used in *run.py* to associate to the user choice in the config file the right function.

    Args:
        value (string): name of the function to reflect particles on the boundaries. Available : specular, diffusive, couette.

    Returns:
        [type]: [description]
    """
    if(value == 'diffusive'):
        return _reflect_particle_diffusive
    elif(value == 'couette'):
        return _couette
    elif(value == 'periodic'):
        return _periodic
    return _reflect_particle_specular # default functions

def _reflect_particle_specular(arr, **kwargs):
    """ Reflect particles given in a specular manner.

    Args:
        arr (np.ndarray): 2D-ndarray of shape *number of particles x 5*. Each particle is describes by : [x, y, vx, vy, vz]

    Returns:
        [np.ndarray]: returns the modified array by default (in most cases, it is useless as the function works in place)
    """
    a, ct, cp = kwargs['directing_vectors'], kwargs['ct'], kwargs['cp']

    # be careful, Theta is the opposite of the angle between the wall and the default coord system.
    k1, k2 = 2*a[:,0]**2-1, 2*a[:,0]*a[:, 1] # 2*ctheta**2-1, 2*ctheta*stheta # TODO : could be saved before computing, this way it gets even faster

    # velocity after the collision
    arr[:,2] = arr[:,2]*k1+ arr[:,3]*k2   # i.e. : vx = vx*k1+vy*k2
    arr[:,3] = - arr[:,3]*k1+arr[:,2]*k2  # i.e. : vy = -vy*k1+vx*k2

    # new position (we could add some scattering which we do not do there)
    arr[:,0] = cp[:,0]+ct*arr[:,2] # new x pos 
    arr[:,1] = cp[:,1]+ct*arr[:,3] # new y pos

    return arr

def _reflect_particle_diffusive(arr, **kwargs):
    """ Reflect particles given in a diffusive manner.

    Args:
        arr (np.ndarray): 2D-ndarray of shape *number of particles x 5*. Each particle is describes by : [x, y, vx, vy, vz]

    Returns:
        [np.ndarray]: returns the modified array by default (in most cases, it is useless as the function works in place)
    """
    # rotating coefficients
    vel_std = gaussian(kwargs['temperature'], kwargs['mass'])
    in_vects = kwargs['normal_vectors']
    quantity = arr.shape[0]
    k1, k2 = in_vects[:,0], in_vects[:,1] # ctheta, stheta

    # initializing velocity in the system : (in_vect, b, in_vect x b), direct system
    drift =  0
    if('drift' in kwargs):
        drift = kwargs['drift']
    tangent_drift = 0
    if('tangent_drift' in kwargs):
        tangent_drift = kwargs['tangent_drift']

    u = vel_std * np.sqrt(-2*np.log((1-np.random.random(size = quantity)))) +  drift
    v = np.random.normal(loc = tangent_drift, scale = vel_std, size = quantity)
    w = np.random.normal(loc = 0, scale = vel_std, size = quantity) # = vz

    # velocity in the right base
    vx = u*k1-v*k2  # i.e. : vx = vx*ctheta + vy*stheta
    vy = v*k1+u*k2  # i.e. : vy = vy*ctheta - vx*stheta
    vel = np.stack((vx, vy, w), axis = 1)

    arr[:, 2:] = vel[:]

    ct, cp = kwargs['ct'], kwargs['cp']

    arr[:,0] = cp[:,0]+ct*arr[:,2] # new x pos 
    arr[:,1] = cp[:,1]+ct*arr[:,3] # new y pos

    return arr

def _periodic(arr, **kwargs): # for only one wall I think no ?
    arr[:,:2] += kwargs['translation']
    return arr

def _couette(arr,**kwargs):
    """ Specific hard coded function for the case of the couette flow with the top boundary being a moving wall (diffusive and drift),
    the bottom one being only a diffusive boundary. While the left and right boundaries follow periodic conditions and are thus considered as specular boundaries.

    Args:
        arr (np.ndarray): 2D-ndarray of shape *number of particles x 5*. Each particle is describes by : [x, y, vx, vy, vz]

    Returns:
        [np.ndarray]: returns the modified array by default (in most cases, it is useless as the function works in place)
    """
    # here we suppose that idxes 0 and 2 are for left and right boundaries and should thus be specular
    # while the top boundary is a diffusive + drift (boundary number 1)
    # and the bottom one is purely diffusive (boundary number 3)
    ct, cp = kwargs['ct'], kwargs['cp']
    idxes_walls = kwargs['index_walls']

    # taking right indexes
    left = np.where(idxes_walls == 0)[0]
    right = np.where(idxes_walls == 2)[0]
    top = np.where(idxes_walls == 1)[0]
    bottom = np.where(idxes_walls == 3)[0]

    # defining the right dicts
    kwargs_left = {
        'translation': np.array([kwargs['tx_left'],0])
    }

    kwargs_right = {
        'translation': np.array([kwargs['tx_right'],0])
    }

    kwargs_top = {
        'temperature':kwargs['temperature'],
        'mass':kwargs['mass'],
        'normal_vectors':kwargs['normal_vectors'][top],
        'tangent_drift': kwargs['drift'],
        'ct':ct[top], 
        'cp':cp[top]
    }
    kwargs_bottom = {
        'temperature':kwargs['temperature'],
        'mass':kwargs['mass'],
        'normal_vectors':kwargs['normal_vectors'][bottom],
        'ct':ct[bottom], 
        'cp':cp[bottom]
    }
    
    arr[left] = _periodic(arr[left], **kwargs_left)
    arr[right] = _periodic(arr[right], **kwargs_right)
    arr[top] = _reflect_particle_diffusive(arr[top], **kwargs_top)
    arr[bottom] = _reflect_particle_diffusive(arr[bottom], **kwargs_bottom)
    
    return arr

# ----------------------------- Wall collision -------------------------- # 

def get_possible_collisions(arr, walls, directing_vectors): # particles are considered as points
    """ Determine if there is a collision between the particule which position, velocity and radius 
    are given in parameters and the wall of index wall_indx. If there is, it compute the time to collision. 
    Note that this algorithms give the number of times a ray, defined by the particle position and future ones 
    (integrating its velocity without changing it), crosses the system boundaries. For simple polygonals geometry, 
    an even number means the particle is outside the boundaries while an odd number means the particle is inside.
    
    We suppose the particule is caracterized by its position (x,y), its velocity (vx, vy) and its radius r.
    The wall is caracterized by its two extremities : p1 = (x1,y1), p2 = (x2,y2) and its normal vector n directed toward the center (such that (p2-p1, n, (p2-p1) x n) is a direct system coordinate).

    The formula which is used is to compute the possible collision time is : 
        t_coll_1/2 = (-a sgn(b) +/- r)/|b| = (-a +/- r)/b
    where :
        * a = -x sin(theta) + y cos(theta)
        * b = -vx sin(theta) + vy cos(theta)
        * theta = sgn(ny) arccos(nx) 
        thus :  cos(theta) = nx
                sin(theta) = ny
        
    Note that theses times give the moment the disk crosses the infinite line formed from the wall,
    not strictly speaking the segment formed by the wall...

    If b = 0 : we consider that there is not collision and return t_coll = np.nan
    
    If b != O : then a necessary condition is to have both t_coll_1 > 0 and t_coll_2 > 0.
    Indeed : * The first time the particule collides, is when its closest point to the wall collides with it. 
             * The second time is for when the furthest point to the wall collided with it.
    In such a case, we have to verify that the disk crossing the line occurs on the portion of the line 
    which is the wall. To do that, we compute the position of the particule at the time of collision and verify that it 
    is on the "wall" segment. If it is we return t_coll = min(t_coll_1, t_coll_2). Else, np.nan.

    Args:
        arr (np.ndarray): array of size *number of particles x 5* where each particle is describe by [x, y, vx, vy, vz].
        walls (np.ndarray) : arrat of size *number of walls x 4*, a wall is described by its two extremities : [x1, y1, x2, y2]
        directing_vectors (np.ndarray) : directing vector of the walls (thus normalized vectors), such that if *a = [ax, ay]* is the directing vector of the wall *w*, ax >= 0 and if ax == 0, then ay > 0. 
    Returns:
        np.ndarray, np.ndarray : returns the collisions times (ct) and positions (cp). Replaces impossible time by np.inf and associated positions by np.nan.
                                 ct shape : number of particles x number of walls
                                 cp shape : number of particles x number of walls x 2
    """
    # since we are determining the past collisions, we have to : velocity -> -velocity
    # a and b
    ctheta, stheta, norm = directing_vectors[:,0], directing_vectors[:, 1], directing_vectors[:, 2] # directing vector of the walls. Normalized !
    p1x, p1y, p2x, p2y = walls[:,0], walls[:,1], walls[:,2], walls[:,3]  #   # np.split(walls, indices_or_sections=4, axis = 1)
    x, y, vx, vy, vz = np.split(arr, indices_or_sections=5, axis = 1) # arr[:,0],  arr[:,1],  arr[:,2],  arr[:,3],  arr[:,4] #

    speed = np.sqrt(vy*vy+vx*vx)

    # split keeps the dimension constant, thus p1x is [number of particles x 1] which allow for the operation later on
    # supposing p2x-p1x > 0
    b = numexpr.evaluate("vx*stheta-vy*ctheta")  # -velocity.x*stheta+velocity.y*ctheta; stheta = p2y-p1y; ctheta = p2x-p1x 
    # b = cos(alpha) where alpha is the angle between the velocity and the normal to the wall.
    a = numexpr.evaluate("(p1x-x)*stheta+(y-p1y)*ctheta")

    # at this point b is 2D and b[i] returns b for all walls for particle i
    
    # possible collision time :
    t_intersect = np.full(shape=b.shape, fill_value=-1.)
    np.divide(-a, b, out=t_intersect, where=b!=0)

    t_intersect = np.where(t_intersect>0,t_intersect,np.inf)
    
    pix = numexpr.evaluate("x-t_intersect*vx")
    piy = numexpr.evaluate("y-t_intersect*vy")
    
    # qty = numexpr.evaluate("((radius+(ctheta*(pix-p1x)+stheta*(piy-p1y))))/(norm+2*radius)")  # dP1.inner(dP2)/(norm_1*norm_1) # norm_1 cant be 0 because wall segments are not on same points.
    qty = numexpr.evaluate("(ctheta*(pix-p1x)+stheta*(piy-p1y))/norm")

    qty = np.where(~np.isnan(qty), qty, -1)

    return np.where((qty >= 0) & (qty <= 1), t_intersect, np.inf), np.moveaxis(np.where((qty >= 0) & (qty <= 1), np.array([pix,piy]), np.nan), 0, -1) # b/speed

def get_relative_indexes(ct, idx_out_walls  ):
    """ Returns a tuple containing:
        - an 1D-boolean-array of size of the number of particles considered here (or ct.shape[0]) where 'True' means the particles is outside the system but did not exited by an "out-wall". 
        - an array containing the indexes of the particles that exited by an "out-wall".
        - an 1D-array, of size the number of particles considered here (i.e. ct.shape[0]), containing the indexes of the walls with which they may have collided first (not all particles are out of the system) 
    Args:
        ct (np.ndarray): 2D-array of shape (nb_particles x nb_walls) containing the intersection times for each particle with each wall
        idx_out_walls (list): List of the walls considered as being "out-walls" - can be empty
        old_colliding_particles (np.ndarray): 1D-boolean-array of size the number of particles, 'True' meaning the particles is outside the system and shoud be reflected back in.
        It is useful since some particles may require several reflections in order to get back in the system. For the first loop turn, it should be set to : np.full((nb_particles), True).

    Returns:
        [tuple]: tuple of np.ndarray
    """
    idxes_walls = np.argmin(ct, axis = 1) # collision time arg min (indicating the possible colliding wall indexes) for each particle
    possible_exiting_particles = np.isin(idxes_walls, idx_out_walls) # if the wall for min collision time is in idx_out_walls (thoses particles should not be processed)
                                                         # True if the particles got out of the system
                                                         # 1D-array of size the number of particles
    # since ct can contains rows of np.inf, we have to account only for the particles with positive non-inf collision time
    possible_colliding_particles = np.count_nonzero(~np.isinf(ct), axis = 1) 
    possible_colliding_particles = ~(np.where(possible_colliding_particles == 0, 1, possible_colliding_particles)%2).astype(bool) # True if it should collide

    # processing only collision that are at the same time true collisions and not with the out wall
    colliding_particles = np.where(possible_colliding_particles &  ~possible_exiting_particles, True, False) # True if should collide, False otherwise - size of the number of particles
    exiting_particles = np.where(possible_colliding_particles & possible_exiting_particles, True, False) # Out of the system and should collide
    idxes_exiting_particles = np.where(exiting_particles)[0] # indexes of particles out of the system (certain may still not be, we need to verify it is not in the system)
                                            # True if exited of the system (and thus we should not compute collisions for theses ones)
    
    # if(old_colliding_particles is not None): # accounting for those who were already reflected back in
    #     c = np.where(old_colliding_particles)[0] # returns the index of the particles that had collided and were already reflected   
    #     old_colliding_particles[c] = old_colliding_particles[c]&colliding_particles # it is updated with the ones particles that after the previous reflection,
                                              # are now inside (meaning some particles will turn from True to False)

    # idxes_out = np.where(old_colliding_particles)[0][idxes_exiting_particles] # indexes of the particle eventually leaving the system after, possibly, several reflections

    return colliding_particles, idxes_exiting_particles, idxes_walls


def get_absolute_indexes(colliding_abs, colliding_relative, relative_idxes_exiting): 
    """ Return the absolute indexes of the colliding particles and exiting the system ones. 
    Indeed, *get_possible_collisions* and *get_relative_indexes* processes the indexes relatively to the array they are given. 
    However, in case of multiple collisions with boundaries for one particle in a time step, a loop over those algorithms is required.
    However, only the particles still needing reflection-in require processing. Thus we only work on the "new particles array" which is obtained with *colliding_abs*.
    This is how we end up with relative indexing compared to the absolute array with all the particles. 
    Thus functions thus convert indexes from relative to absolute. For *colliding_relative*, it is done in place. For *relative_idxes_exiting*,
    it is returned.

    Args:
        colliding_abs (np.ndarray): array of booleans of size *number of particles*. A True value means the particle is still outside the system.
        colliding_relative (np.ndarray): array of booleans of size *number of True value in colliding_abs* giving for every still-outside particles, 
                                         if after the new turn, the particle is still outside (True) or was indeed reflected back inside (False).
        relative_idxes_exiting (np.ndarray): array of int, each value is the relative index of an exiting particles in the relative array of particles obtained by arr[colliding_abs].

    Returns:
        [np.ndarray]: arrays of int, each value is the absolute index of the exiting particles in the array of particles
    """
    c =  np.where(colliding_abs)[0]
    colliding_abs[c] = colliding_abs[c] & colliding_relative # in place
    return c[relative_idxes_exiting]

def reflect_back_in(arr, absolute_colliding_bool, relative_colliding_bool, generic_args, reflect_fn, user_generic_args, user_defined_args): # ct : collision time, cp : collision position
    """ Reflect particles given by *arr[absolute_colliding_bool]* back in the system. This is a INPLACE function.
    It will automatically compute the right args depending on the choice in *user_generic_args* and *user_defined_args*. 

    Args:
        arr (np.ndarray): 2D-ndarray of size *number of particles x 5*, a particle is [x,y,vx,vy,vz].
        absolute_colliding_bool (np.ndarray): array of booleans of size *number of particles*. A True value means the particle is still outside the system.
        relative_colliding_bool (np.ndarray): array of booleans of size *number of True value in absolute_colliding_bool* giving for every still-outside particles, 
                                              if after the new turn, the particle is still outside (True) or was indeed reflected back inside (False).
                                              Arguments given in *generic_args* rely on *relative_colliding_bool*, as they were process this turn and have the size 
                                              of *number of True value in absolute_colliding_bool*
        generic_args (dict): a dictionnary containing :
                                index_walls () :
                                cp () :
                                ct () :
                                directing_vectors () :
                                normal_vectors () :
                                mass () :
        reflect_fn (function): the reflection function. Default ones are given at the top of this file.
        user_generic_args (list): a list of keys (strings) related to *generic_args*. Associated values in *generic_args* will then be processed 
                                  in agreement with *relative_colliding_bool*. The computations being somewhat expensive, this is why the user can select only those he needs.
        user_defined_args (dict): this a dict with additionnal parameters for the functions (which are not generic one that can, for most of them, only be computed in the simulations).
                                Those args, if the code is used with the command line, are given as float in the cfg file.
    """
    # no particles to reflect back in
    if(np.count_nonzero(relative_colliding_bool)==0) :
        return

    indexes_walls = generic_args['index_walls']
    # Next is the selection of the args and processing (we need to take only the values needed !)
    if('cp' in user_generic_args):
        cp = np.take_along_axis(generic_args['cp'], indexes_walls[:,None, None], axis = 1)[relative_colliding_bool, :].squeeze(axis=1) # cp is a 2D-array (containg position (x, y))
        user_defined_args['cp'] = cp
    if('ct' in user_generic_args):
        ct = np.take_along_axis(generic_args['ct'], indexes_walls[:,None], axis = 1)[relative_colliding_bool, :].squeeze(axis=1) # ct is a 1D-array (containing times)
        user_defined_args['ct'] = ct
    if('directing_vectors' in user_generic_args):
        directing_vectors = np.take_along_axis(generic_args['directing_vectors'], indexes_walls[:,None], axis = 0)[relative_colliding_bool, :] # walls_directing_vectors is a 2D-array of 2D-directing vectors
        user_defined_args['directing_vectors'] = directing_vectors
    if('normal_vectors' in user_generic_args):
        normal_vectors = np.take_along_axis(generic_args['normal_vectors'], indexes_walls[:,None], axis = 0)[relative_colliding_bool, :] # walls_directing_vectors is a 2D-array of 2D-directing vectors
        user_defined_args['normal_vectors'] = normal_vectors
    if('index_walls' in user_generic_args):
        index_walls = indexes_walls[relative_colliding_bool] # size becomes less than arr
        user_defined_args['index_walls'] = index_walls
    if('mass' in user_generic_args):
        user_defined_args['mass'] = generic_args['mass']

    arr[absolute_colliding_bool,:] = reflect_fn(arr[absolute_colliding_bool,:], **user_defined_args)