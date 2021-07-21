import numpy as np
from random import random
import numpy.ma as ma
import numexpr

# ------------------------- reflection default functions ------------------ #

def reflection_functions_dispatcher(value):
    if(value == 'diffusive'):
        print('Diffusive is not implemented yet')
        return _reflect_particle_diffusive
    return _reflect_particle_specular # default functions

def _reflect_particle_specular(arr, idxes_walls, a, ct, cp): # _specular
    # be careful, Theta is the opposite of the angle between the wall and the default coord system.
    k1, k2 = 2*a[:,0]**2-1, 2*a[:,0]*a[:, 1] # 2*ctheta**2-1, 2*ctheta*stheta # TODO : could be saved before computing, this way it gets even faster

    # velocity after the collision
    arr[:,2] = arr[:,2]*k1+ arr[:,3]*k2   # i.e. : vx = vx*k1+vy*k2
    arr[:,3] = - arr[:,3]*k1+arr[:,2]*k2  # i.e. : vy = -vy*k1+vx*k2

    # new position (we could add some scattering which we do not do there)
    arr[:,0] = cp[:,0]+ct*arr[:,2] # new x pos 
    arr[:,1] = cp[:,1]+ct*arr[:,3] # new y pos

    return arr

def _reflect_particle_diffusive(arr, a, ct, cp):
    return arr
# ----------------------------- Wall collision -------------------------- #

# TODO : optimization should be possible for this function 
# TODO : This function is doing too much and should be split, especially since walls-dependant collisions needs to be added (example : moving wall) 
# and since we also need possible various reflection depending on the species (diffusive vs specular)
# all that required the :
    # - modification of the way we call the reflection function (which is for now always _reflect_particle)
    # - adding the index of the wall to be taken into account in the way the reflection is performed (this should be also changed trough _reflect_particle)

# TODO : refaire la suite en faisant bien la différence entre les différentes bases et la définition de chaque angle selon le système de coordonnées choisis...
# parce qu'en pratique j'ai mal appelé mes angles ...
# En effet, je cherche l'angle que j'appelle "radial", qui est celui formé par la vélocité et le vecteur directeur du mur dans le plan 2D global (x,y)
# sauf que dans le repère du mur, on est plutôt dans le plan 2D (y',z'), avec z' la normale sortante du mur vers l'intérieur du système... ce qui complique les appelations...
# donc en fait dans le repère initial, je cherche par conséquent plutôt l'angle azimutal, qui est l'angle radial dans le repère du mur si on appelle z' la normale sortante
# mais qui est toujours l'angle azimutal si je garde un repère du type (x,y,z') qui est ce que j'ai fait jusqu'à présent
# on peut dire que c'est l'angle radial lorsque la normal sortante est y dans le repère de base
# arr is number of particles x (pos_end_idx+3) - we use 3D velocity and 2D pos by default
# walls is a 2D array consisting wall = np.array([x1, y1, x2, y2]) (for now)
# these walls have been stored such that x1 < x2, in case x1 == x2, y1 < y2.
# and thus a = np.array([x2-x1, y2-y1])/norm (it has been normalized)
# therefore, theta, defined as in : <a|x> = cos(theta), yields : 
    # - theta is in (-pi/2, pi/2]
    # - cos(theta) is in (0,1] (however not on its bijective interval)
    # - sin(theta) is in [-1, 1] (bijective, and should therefore be used to get back to theta)
# This single-handedly allows any 2D-reflection where we are only concerned with the radial angle (also called the 'colatitude' in french)
# since we can then rotate any velocity vector in the wall-coordinates-system B'.
# However, we would also like to have the azimuthal angle to perform proper 3D reflection that are useful for any reflection other than purely specular ones
# as we would have a function taking the incident azimutal angle and returning the reflected one for example
# (or rather something like : Ef, Tf, Pf = f(Ei, Ti, Pf), where E, T and P are respectively the energy, and radial and azimutal angles)

# in order to have both, we need for a given velocity vector v = (vx, vy, vz) ~ (rho, Ti, Pi) with Ti in [-pi/2,pi/2] (in theory it's algebric in [0,pi/2]) and Pi in [0, pi] (instead of [0,2pi]):
    # - radial angle with walls : Ti' = Ti - theta
    # - azimutal : Pi' = Pi
# the tricky part is : how do you compute Ti and Pi ?
    # - arccos()
def get_possible_collisions(arr, walls, a): # particles are considered as points
    # TODO : je pense que je devrais passer tout ça sur 100% numpy et pas numexpr. En réalité, je risque pas d'utiliser souvent pour plusieurs particles (que celles qui sont sorties du système).
    # boucle sur les particules a priori.
    # En fait cet algo donne aussi la présence de la particule dans le système, puisque si on est dans le système, alors on a un nombre impair de murs avec lesquels on peut collisionner.
    """ Determine if there is a collision between the particule which position, velocity and radius 
    are given in parameters and the wall of index wall_indx.
    If there is, it compute the time to collision and update the events table.
    
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
        part_indx (int): index of the particule in self.particules
        position (MyVector): position of the particule
        velocity (MyVector): velocity of the particule
        radius (float): radius of the particule
        wall_indx (int): index of the wall in self.walls

    Returns:
        int, MyVector: the time before the wall and particule collides. Return np.nan is no collision is possible. 
    """
    # since we are determining for past collisions, we have to velocity -> -velocity
    # a and b
    ctheta, stheta, norm = a[:,0], a[:, 1], a[:, 2] # directing vector of the walls. Normalized !
    p1x, p1y, p2x, p2y = walls[:,0], walls[:,1], walls[:,2], walls[:,3]  #   # np.split(walls, indices_or_sections=4, axis = 1)
    x, y, vx, vy, vz = np.split(arr, indices_or_sections=5, axis = 1) # arr[:,0],  arr[:,1],  arr[:,2],  arr[:,3],  arr[:,4] #

    speed = np.sqrt(vy*vy+vx*vx)

    # split keeps the dimension constant, thus p1x is [number of particles x 1] which allow for the operation later on
    # supposing p2x-p1x > 0
    b = numexpr.evaluate("vx*stheta-vy*ctheta")  # -velocity.x*stheta+velocity.y*ctheta; stheta = p2y-p1y; ctheta = p2x-p1x 
    # b = cos(alpha) where alpha is the angle between the velocity and the normal to the wall.
    a_prime = numexpr.evaluate("(p1x-x)*stheta+(y-p1y)*ctheta")

    # at this point b is 2D and b[i] returns b for all walls for particle i
    
    # possible collision time :
    t_intersect = np.full(shape=b.shape, fill_value=-1.)
    np.divide(-a_prime, b, out=t_intersect, where=b!=0)

    t_intersect = np.where(t_intersect>0,t_intersect,np.inf)
    
    pix = numexpr.evaluate("x-t_intersect*vx")
    piy = numexpr.evaluate("y-t_intersect*vy")
    
    # qty = numexpr.evaluate("((radius+(ctheta*(pix-p1x)+stheta*(piy-p1y))))/(norm+2*radius)")  # dP1.inner(dP2)/(norm_1*norm_1) # norm_1 cant be 0 because wall segments are not on same points.
    qty = numexpr.evaluate("(ctheta*(pix-p1x)+stheta*(piy-p1y))/norm")

    qty = np.where(~np.isnan(qty), qty, -1)

    return np.where((qty >= 0) & (qty <= 1), t_intersect, np.inf), np.moveaxis(np.where((qty >= 0) & (qty <= 1), np.array([pix,piy]), np.nan), 0, -1) # b/speed

def get_indexes(ct, idx_out_walls, old_colliding_particles):
    """ Returns a tuple containing:
        - an 1D-boolean-array of size the number of particles (or ct.shape[0]) where 'True' means the particles is outside the system but did not exited by an "out-wall". 
        - an array containing the indexes of the particles that exited by an "out-wall".
        - an 1D-array, of size the number of particles, containing the indexes of the walls with which they may have collided first (not all particles are out of the system) 
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
                                            # True if out of the system (and thus we should not compute collisions for theses ones)

    if(old_colliding_particles is not None): # accounting for those who were already reflected back in
        c = np.where(old_colliding_particles)[0] # returns the index of the particles that had collided and were already reflected   
        old_colliding_particles[c] = old_colliding_particles[c]&colliding_particles # it is updated with the ones particles that after the previous reflection,
                                                 # are now inside (meaning some particles will turn from True to False)
        idxes_out = c[idxes_exiting_particles] # taking the indexes of the particles stil outside 
        colliding_particles = old_colliding_particles # c2 is sent back in this function for the next iteration as 'old_colliding_particles'

    return colliding_particles, idxes_out, idxes_walls

def reflect_back_in(arr, colliding_particles, idxes_walls, walls_directing_vectors, ct, cp, reflect_fn): # ct : collision time, cp : collision position
    # INPLACE
    # no particles to reflect back in
    if(np.count_nonzero(colliding_particles)==0) :
        return

    cp_ = np.take_along_axis(cp, idxes_walls[:,None, None], axis = 1)[colliding_particles, :].squeeze(axis=1) # cp is a 2D-array (containg position (x, y))
    ct_ = np.take_along_axis(ct, idxes_walls[:,None], axis = 1)[colliding_particles, :].squeeze(axis=1) # ct is a 1D-array (containing times)
    walls_directing_vectors_ = np.take_along_axis(walls_directing_vectors, idxes_walls[:,None], axis = 0)[colliding_particles, :] # walls_directing_vectors is a 2D-array of 2D-directing vectors
    idxes_walls_ = idxes_walls[colliding_particles] # np.take_along_axis(idxes_walls, idxes_walls[:,None], axis = 1)[colliding_particles, :].squeeze(axis=1) # idxes_walls is a 1D-array of indexes
    arr[colliding_particles,:] = reflect_fn(arr[colliding_particles,:], idxes_walls_, walls_directing_vectors_, ct_, cp_)


# --- bins - saved --- #

# def make_collisions_out_walls(arr, a, ct, cp, idx_out_walls, old_colliding_particles = None, cos_alpha = None): # ct : collision time, cp : collision position
#     idxes_walls = np.argmin(ct, axis = 1) # collision time arg min (indicating the possible colliding wall indexes) for each particle
#     possible_exiting_particles = np.isin(idxes_walls, idx_out_walls) # if the wall for min collision time is in idx_out_walls (thoses particles should not be processed)
#                                                          # True if the particles got out of the system
#                                                          # 1D-array of size the number of particles
#     # since ct can contains rows of np.inf, we have to account only for the particles with positive non-inf collision time 
#     possible_colliding_particles = np.count_nonzero(~np.isinf(ct), axis = 1) 
#     possible_colliding_particles = ~(np.where(possible_colliding_particles == 0, 1, possible_colliding_particles)%2).astype(bool) # True if it should collide

#     # processing only collision that are at the same time true collisions and not with the out wall
#     colliding_particles = np.where(possible_colliding_particles &  ~possible_exiting_particles, True, False) # True if should collide, False otherwise - size of the number of particles
#     exiting_particles = np.where(possible_colliding_particles & possible_exiting_particles, True, False) # Out of the system and should collide
#     idxes_exiting_particles = np.where(exiting_particles)[0] # indexes of particles out of the system (certain may still not be, we need to verify it is not in the system)
#                                             # True if out of the system (and thus we should not compute collisions for theses ones)

#     if(old_colliding_particles is not None): # accounting for those who were already reflected back in
#         c = np.where(old_colliding_particles)[0] # returns the index of the particles that had collided and were already reflected   
#         old_colliding_particles[c] = old_colliding_particles[c]&colliding_particles # it is updated with the ones particles that after the previous reflection,
#                                                  # are now inside (meaning some particles will turn from True to False)
#         idxes_out = c[idxes_exiting_particles] # taking the indexes of the particles stil outside 
#         c2 = old_colliding_particles # c2 is sent back in this function for the next iteration as 'old_colliding_particles'
#     else:
#         c2 = colliding_particles

#     if(np.count_nonzero(colliding_particles)==0): # np.sum(count, where = count == True) (only in python 3.9)
#         if(cos_alpha is None):
#             return c2, idxes_out
#         else :
#             return c2, idxes_out, np.take_along_axis(cos_alpha, idxes_walls[:,None], axis = 1)[colliding_particles, :].squeeze(axis=1)

#     cp_ = np.take_along_axis(cp, idxes_walls[:,None, None], axis = 1)[colliding_particles, :].squeeze(axis=1)
#     ct_ = np.take_along_axis(ct, idxes_walls[:,None], axis = 1)[colliding_particles, :].squeeze(axis=1)
#     a_ = np.take_along_axis(a, idxes_walls[:,None], axis = 0)[colliding_particles, :] 
    
#     arr[c2,:] = _reflect_particle(arr[c2,:], a_, ct_, cp_)

#     if(cos_alpha is None):
#         return c2, idxes_out # the indexes in arr of the particles that got out of the system by out_walls
#     else:
#         return c2, idxes_out, np.take_along_axis(cos_alpha, idxes_walls[:,None], axis = 1)[colliding_particles, :].squeeze(axis=1)
