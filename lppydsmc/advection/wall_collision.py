import numpy as np
from random import random
import numpy.ma as ma
import numexpr
# ----------------------------- Wall collision -------------------------- #

# NOTE : THE THREE FOLLOWING FUNCTIONS ARE INPLACE

def make_collisions(arr, a, ct, cp, old_count = None): # ct : collision time, cp : collision position
    count = np.count_nonzero(~np.isinf(ct), axis = 1) # problem when there is no collision to process with no wall whatsoever. Because 0 counts as requiring a collision to be processed...
    count = np.where(count == 0, -1, count)%2 # not super optimal I think.
    idxes = np.argmin(ct, axis = 1)

    # count = [0, 0, 0, 1] for example, 0 if outside, 1 if inside.

    # with loop:
    for k, c in enumerate(count) :
        if(c==0):
            # reflect
            idx = idxes[k]
            arr[[k],:] = _reflect_particle(arr[[k],:], a[[idx],:], ct[[k], [idx]], cp[[k], [idx]]) # arr, a, ct, cp
    return count

def make_collisions_vectorized(arr, a, ct, cp, old_count = None): # ct : collision time, cp : collision position
    idxes = np.argmin(ct, axis = 1)
    count = np.count_nonzero(~np.isinf(ct), axis = 1) # problem when there is no collision to process with no wall whatsoever. Because 0 counts as requiring a collision to be processed...
    count = ~(np.where(count == 0, -1, count)%2).astype(bool) # not super optimal I think.
    # ic(np.sum(count, where = count == True))
    
    #count = ~count.astype(bool)
    # ct = np.where(np.isinf(ct), 0, ct)
    # count = [0, 0, 0, 1] for example, 0 if outside, 1 if inside.
    # ---- Overhead to avoid the python loop (and multiple call to a function) ---:
        # ct and cp mush be shrink in dimension, from (number of particles x number of walls) to (number of particles)
        # a must be changes to a vector [number of particles], where is line is the required wall 
        # all this depends on idxes which is of size [number of particles]
    cp_ = np.take_along_axis(cp, idxes[:,None, None], axis = 1)[count, :].squeeze(axis=1)
    ct_ = np.take_along_axis(ct, idxes[:,None], axis = 1)[count, :].squeeze(axis=1)
    a_ = np.take_along_axis(a, idxes[:,None], axis = 0)[count, :]
    arr[count,:] = _reflect_particle(arr[count,:], a_, ct_, cp_)
    return count

def make_collisions_out_walls(arr, a, ct, cp, idx_out_walls, old_count = None): # ct : collision time, cp : collision position
    # basically the same function as make_collisions_vectorized, only we add some conditions
    idxes = np.argmin(ct, axis = 1)
    idxes_out_bool = np.isin(idxes, idx_out_walls) # if the wall for min collision time is in idx_out_walls 
    
    # getting actual collision
    count = np.count_nonzero(~np.isinf(ct), axis = 1) 
    count = ~(np.where(count == 0, 1, count)%2).astype(bool) # not super optimal I think.
                                                            # True if it should collide

    

    # processing only collision that are at the same time true collisions and not with the out wall
    # TODO : clean it
    count_ = np.where(count &  ~idxes_out_bool, True, False)
    idxes_out_bool = np.where(count & idxes_out_bool, True, False) # Out of the system and should collide
    idxes_out = np.where(idxes_out_bool)[0] # indexes of particles out of the system (certain may still not be, we need to verify it is not in the system)
                                            # True if out of the system (and thus we should not compute collisions for theses ones)
    count = count_

    if(old_count is not None):
        c = np.where(old_count)[0]
        old_count[c] = old_count[c]&count
        idxes_out = c[idxes_out]
        c2 = old_count
    else:
        c2 = count

    if(np.sum(count, where = count == True)==0):
        return c2, idxes_out
    
    cp_ = np.take_along_axis(cp, idxes[:,None, None], axis = 1)[count, :].squeeze(axis=1)
    ct_ = np.take_along_axis(ct, idxes[:,None], axis = 1)[count, :].squeeze(axis=1)
    a_ = np.take_along_axis(a, idxes[:,None], axis = 0)[count, :]
    arr[c2,:] = _reflect_particle(arr[c2,:], a_, ct_, cp_)

    return c2, idxes_out # the indexes in arr of the particles that got out of the system by out_walls

pos_end_idx = 2

# arr is number of particles x (pos_end_idx+3) - we use 3D velocity and 2D pos by default
# walls is a 2D array consisting wall = np.array([x1, y1, x2, y2]) (for now)
# these walls have been stored such that x1 < x2, in case x1 == x2, y1 < y2.

def handler_wall_collision(arr, walls, a, radius): 
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
    # split keeps the dimension constant, thus p1x is [number of particles x 1] which allow for the operation later on
    # supposing p2x-p1x > 0
    b = numexpr.evaluate("vx*stheta-vy*ctheta")  # -velocity.x*stheta+velocity.y*ctheta; stheta = p2y-p1y; ctheta = p2x-p1x
    a_prime = numexpr.evaluate("(p1x-x)*stheta+(y-p1y)*ctheta")

    # at this point b is 2D and b[i] returns b for all walls for particle i
    
    # possible collision time :
    t_coll_1 = np.full(shape=b.shape, fill_value=-1.)
    t_coll_2 = np.full(shape=b.shape, fill_value=-1.)
    
    np.divide((-a_prime-radius), b, out=t_coll_1, where=b!=0)
    np.divide((-a_prime+radius), b, out=t_coll_2, where=b!=0)

    # ic(t_coll_1)
    # ic(t_coll_2)

    t_intersect = np.full(shape=b.shape, fill_value=np.inf)
    t_intersect = np.maximum(t_coll_1, t_coll_2 , where = t_coll_2 + t_coll_1 > 0, out = t_intersect) # ((t_coll_2 > 0) | (t_coll_1>0))
    
    picx = numexpr.evaluate("x-0.5*(t_coll_1+t_coll_2)*vx")
    picy = numexpr.evaluate("y-0.5*(t_coll_1+t_coll_2)*vy")
    pix = numexpr.evaluate("x-t_intersect*vx")
    piy = numexpr.evaluate("y-t_intersect*vy")
    # qty = numexpr.evaluate("((radius+(ctheta*(pix-p1x)+stheta*(piy-p1y))))/(norm+2*radius)")  # dP1.inner(dP2)/(norm_1*norm_1) # norm_1 cant be 0 because wall segments are not on same points.
    qty = numexpr.evaluate("(ctheta*(picx-p1x)+stheta*(picy-p1y))/norm")
    # ic(qty)
    qty = np.where(~np.isnan(qty), qty, -1)
    return np.where((qty >= 0) & (qty <= 1), t_intersect, np.inf), np.moveaxis(np.where((qty >= 0) & (qty <= 1), np.array([pix,piy]), np.nan), 0, -1)

def handler_wall_collision_point(arr, walls, a, deal_with_corner = False): # particles are considered as points
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
    # split keeps the dimension constant, thus p1x is [number of particles x 1] which allow for the operation later on
    # supposing p2x-p1x > 0
    b = numexpr.evaluate("vx*stheta-vy*ctheta")  # -velocity.x*stheta+velocity.y*ctheta; stheta = p2y-p1y; ctheta = p2x-p1x
    a_prime = numexpr.evaluate("(p1x-x)*stheta+(y-p1y)*ctheta")

    # at this point b is 2D and b[i] returns b for all walls for particle i
    
    # possible collision time :
    t_intersect = np.full(shape=b.shape, fill_value=-1.)
    np.divide(-a_prime, b, out=t_intersect, where=b!=0)

    t_intersect = np.where(t_intersect>0,t_intersect,np.inf)
    # ic(t_intersect)

    pix = numexpr.evaluate("x-t_intersect*vx")
    piy = numexpr.evaluate("y-t_intersect*vy")
    
    # ic(pix[0])
    # ic(piy[0])

    # ic(p1x[0])
    # ic(ctheta[0])
    # ic(norm[0])
    # qty = numexpr.evaluate("((radius+(ctheta*(pix-p1x)+stheta*(piy-p1y))))/(norm+2*radius)")  # dP1.inner(dP2)/(norm_1*norm_1) # norm_1 cant be 0 because wall segments are not on same points.
    qty = numexpr.evaluate("(ctheta*(pix-p1x)+stheta*(piy-p1y))/norm")
    
    # ic(qty)

    qty = np.where(~np.isnan(qty), qty, -1)

    return np.where((qty >= 0) & (qty <= 1), t_intersect, np.inf), np.moveaxis(np.where((qty >= 0) & (qty <= 1), np.array([pix,piy]), np.nan), 0, -1)

def _reflect_particle(arr, a, ct, cp):
    # be careful, Theta is the opposite of the angle between the wall and the default coord system.
    k1, k2 = 2*a[:,0]**2-1, 2*a[:,0]*a[:, 1] # 2*ctheta**2-1, 2*ctheta*stheta # TODO : could be saved before computing, this way it gets even faster

    # velocity after the collision
    arr[:,2] = arr[:,2]*k1+ arr[:,3]*k2   # i.e. : vx = vx*k1+vy*k2
    arr[:,3] = - arr[:,3]*k1+arr[:,2]*k2  # i.e. : vy = -vy*k1+vx*k2

    # new position (we could add some scattering which we do not do there)
    arr[:,0] = cp[:,0]+ct*arr[:,2] # new x pos 
    arr[:,1] = cp[:,1]+ct*arr[:,3] # new y pos

    return arr

# very useless
def deal_with_corner(ct):
    # very inefficient
    for k in range(ct.shape[0]):
        ct_, indexes = np.unique(ct[k], return_index = True)
        if(ct_.shape != ct[k].shape):
            ct[k, :] = np.inf
            ct[k, indexes] = ct_