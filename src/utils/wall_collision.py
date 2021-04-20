 
from src.utils.segment import Segment
from src.utils.vector import Vector

import numpy as np
from random import random

# ----------------------------- Wall collision -------------------------- #
def _collision_with_wall(pos, velocity, radius, segments, strategy = 'future'):
    if(strategy == 'past'):
        velocity = -1.0*velocity
    min_time, idx_segment, min_pos_intersect = 1e15 ,-1, None

    for i, segment in enumerate(segments):
        t_coll, pos_intersect = _handler_wall_collision(pos, velocity, radius, segment.get_p1(), segment.get_p2(), strategy) # should add  -1.0*velocity if we are checking in the past...
        #print(f'{i} : t_coll = {t_coll} ; intersect = {pos_intersect}')
        if(t_coll<min_time):
            idx_segment = i
            min_time = t_coll
            min_pos_intersect = pos_intersect
    return min_time, idx_segment, min_pos_intersect

def _handler_wall_collision(position, velocity, radius, p1, p2, strategy = 'future'):
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
    # p index of the part

    # angle
    #theta = np.arccos(n.x) # np.sign(n.y) 

    # a and b
    dp = p2-p1
    if(dp.x > 0):
        stheta, ctheta = dp.y, dp.x
    else:
        stheta, ctheta = -dp.y, dp.x

    b = -velocity.x*stheta+velocity.y*ctheta

    if b == 0.0 :
        return np.nan, None # TODO : should we add a tolerance ? It will never be equals to zero exactly...
    
    a = -position.x*stheta+position.y*ctheta
    
    # new position of the wall in the new base
    y1_new_base = -p1.x*stheta+p1.y*ctheta

    a_prime = a-y1_new_base

    # possible collision time :
    t_coll_1 = (-a_prime-2*radius)/b
    t_coll_2 = (-a_prime+2*radius)/b
    if(t_coll_1 < 0 or t_coll_2 <0):
        return np.nan, None

    if(strategy == 'past'): # stategy for finding a past collision
        t_intersect = max(t_coll_1, t_coll_2)
    elif(strategy=='future'): # stategy for finding a future collision
        t_intersect = min(t_coll_1, t_coll_2)
    else:
        t_intersect = min(t_coll_1, t_coll_2)
        print(f'Please choose between *past* (past collision) and *future* (future collision) for the strategy. You chose {strategy}. Default to *future*.')
        
    if(t_intersect > 0):
        pos_intersect = position + t_intersect * velocity

        # the reason why were are not using pos_intersect.norm is that it should be a 3D vector.
        dP1, dP2, dP3 = p2-p1, \
            pos_intersect-p1, p2-pos_intersect 
        norm_1 = dP1.norm()

        qty=dP1.inner(dP2)/(norm_1*norm_1) # norm_1 cant be 0 because wall segments are not on same points.
        if(qty < 1 and qty > 0):
            return t_intersect, pos_intersect

    return np.nan, None

def _reflect_particle(velocity, time, dp, pos_intersect):

    # SPEED reflection
    
    # angle
    if(dp.x<0):
        dp = -1.0*dp
        
    theta = float(np.sign(dp.y)*np.arccos(dp.x)) # angle between the directing vector and (0,1)
    
    # theta must be in degree .... 
    theta = theta*180/np.pi

    # old velocity
    old_velocity = Vector(velocity.x, velocity.y) # in 2D
    intermediary_velocity = old_velocity.rotate(-theta)
    intermediary_velocity_2 = Vector(intermediary_velocity.x, -intermediary_velocity.y)
    new_velocity_2d = intermediary_velocity_2.rotate(theta)
    
    # SETTING new velocity 
    new_velocity_3d = Vector(new_velocity_2d.x, new_velocity_2d.y, velocity.z)
    new_velocity = new_velocity_3d

    # POSITION reflection
    new_pos = Vector(pos_intersect.x, pos_intersect.y, pos_intersect.z) + time*new_velocity_3d # (0.1*(0.5-random())+1)*

    return new_pos, new_velocity