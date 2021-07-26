# Imports
import numpy as np

# local modules
from .creator import SystemCreator

# -------------------- This Module -------------------------- #
"""
This module gives an aid for defining basic systems :
    - thruster (with the geometry you want). Call : thruster_three_grids_system, thruster_system
    - rectangle. Call : system_rectangle.
    - rectangle with a cylinder obstacle. Call : cylinder_system.
"""

def points_to_segments(points):
    """Convert a list of points, given in clockwise order compared to the inside of the system to a list of segments.
    The last point being linked to the first one. 

    Args:
        points (list): list of lists of size 2

    Returns:
        [np.ndarray]: 2D-array of segments - each row is [x1,y1,x2,y2].
    """
    nb_of_points = len(points)
    points = np.array(points)
    first, last = points[0], points[-1]
    segments = np.concatenate((points[:nb_of_points-1],points[1:]), axis = 1)
    segments = np.concatenate((segments, np.expand_dims(np.concatenate((last,first)), axis = 0)), axis = 0)
    return segments
    # --------------------- Utils functions -------------------- #

def rectangle_(w,l, offset=np.array([0,0])):
    """[summary]

    Args:
        w ([type]): [description]
        l ([type]): [description]
        offset ([type], optional): [description]. Defaults to np.array([0,0]).

    Returns:
        [type]: [description]
    """
    # top left point is p1 and then its anti-trigo rotation
    p1 = np.array([0,l])+offset
    p2 = np.array([w,l])+offset
    p3 = np.array([w,0])+offset
    p4 = np.array([0,0])+offset
    return p1,p2,p3,p4

    # --------------------- Rectangle system ----------------- #

def system_rectangle(l_x, l_y, offsets = np.array([0,0])):
    # segment order : top, right, bottom, left
    points = np.array(rectangle_(l_x, l_y, offsets))
    segments = np.concatenate((points[:3], points[1:]), axis = 1)
    segments = np.concatenate((segments, np.expand_dims(np.concatenate((points[-1],points[0])),axis = 0)), axis = 0)
    return SystemCreator(segments, [1,3]), 3

    # --------------------- Thruster system ----------------- #

def thruster_points(w_in, l_in, w_1, l_1, l_int, w_2, l_2, w_out, l_out, offsets):
    p2, p3, p20, p1 = rectangle_(l_in, w_in, offset = offsets)
    p4, p5, p18, p19 = rectangle_(l_1, w_1, offset = offsets+np.array([l_in,0.5*(w_in-w_1)]))
    p6, p7, p16, p17 = rectangle_(l_int, w_in, offset = offsets+np.array([l_1+l_in, 0]))
    p8, p9, p14, p15 = rectangle_(l_2,w_2, offset = offsets+np.array([l_in+l_1+l_int,0.5*(w_in-w_2)]))
    p10, p11, p12, p13 = rectangle_(l_out, w_out, offset = offsets+np.array([l_in+l_1+l_int+l_2,0.5*(w_in-w_out)]))
    points = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20])
    return points

def thruster_three_grids_points(w_in, l_in, w_1, l_1, l_int, w_2, l_2, l_int_2, w_3, l_3, w_out, l_out, offsets):
    p2, p3, p28, p1 = rectangle_(l_in, w_in, offset = offsets)
    p4, p5, p26, p27 = rectangle_(l_1, w_1, offset = offsets+np.array([l_in,0.5*(w_in-w_1)]))
    p6, p7, p24, p25 = rectangle_(l_int, w_in, offset = offsets+np.array([l_1+l_in, 0]))
    p8, p9, p22, p23 = rectangle_(l_2, w_2, offset = offsets+np.array([l_in+l_1+l_int,0.5*(w_in-w_2)]))
    p10, p11, p20, p21 = rectangle_(l_int_2, w_in, offset = offsets+np.array([l_in+l_1+l_int+l_2, 0]))
    p12, p13, p18, p19 = rectangle_(l_3, w_3, offset = offsets+np.array([l_in+l_1+l_int+l_2+l_int_2,0.5*(w_in-w_3)]))
    p14, p15, p16, p17 = rectangle_(l_out, w_out, offset = offsets+np.array([l_in+l_1+l_int+l_2+l_int_2+l_3,0.5*(w_in-w_out)]))
    points = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28])
    return points
    
def thruster(w_in, l_in, w_1, l_1, l_int, w_2, l_2, w_out, l_out, offsets = np.array([0,0])):
    # hypothesis : w_int = w_in
    # returns an array with the walls for the thruster
    # not optimized but we prioritize the clarity here
    points = thruster_points(w_in, l_in, w_1, l_1, l_int, w_2, l_2, w_out, l_out, offsets =offsets)
    p1, p20 = points[0], points[-1] 
    segments = np.concatenate((points[1:],points[:19]), axis = 1)
    segments = np.concatenate((segments, np.expand_dims(np.concatenate((p20,p1)), axis = 0)), axis = 0)
    # sorting is realized when the array is created per the SystemCreator. No need to worry at this point.
    return segments # system, idx_out_walls, idx_in_wall

def thruster_three_grids(w_in, l_in, w_1, l_1, l_int, w_2, l_2, l_int_2, w_3, l_3, w_out, l_out, offsets = np.array([0,0])):
    # hypothesis : w_int = w_in
    # returns an array with the walls for the thruster
    # not optimized but we prioritize the clarity here
    points = thruster_three_grids_points(w_in, l_in, w_1, l_1, l_int, w_2, l_2, l_int_2, w_3, l_3, w_out, l_out, offsets =offsets)
    p1, p28 = points[0], points[-1]
    segments = np.concatenate((points[1:],points[:27]), axis = 1)
    segments = np.concatenate((segments, np.expand_dims(np.concatenate((p28,p1)), axis = 0)), axis = 0)
    # sorting is realized when the array is created per the SystemCreator. No need to worry at this point.
    return segments # system, idx_out_walls, idx_in_wall

def thruster_three_grids_system(w_in, l_in, w_1, l_1, l_int, w_2, l_2, l_int_2, w_3, l_3, w_out, l_out, offsets = np.array([0,0])):
    segments = thruster_three_grids(w_in, l_in, w_1, l_1, l_int, w_2, l_2, l_int_2, w_3, l_3, w_out, l_out, offsets = offsets)
    return SystemCreator(segments, [0, 13, 14, 15]), 0 
    
def thruster_system(w_in, l_in, w_1, l_1, l_int, w_2, l_2, w_out, l_out, offsets = np.array([0,0])):
    segments = thruster(w_in, l_in, w_1, l_1, l_int, w_2, l_2, w_out, l_out, offsets = offsets)
    return SystemCreator(segments,  [0, 10, 9, 11]), 0 # system, idx_out_walls, idx_in_wall
    # 4 out walls : in wall, + P10-P11 + P11 - P12 + P12 - P13

    # --------------------- Cylinder system ----------------- #

def cylinder_system(res, l_x, l_y, c_x, c_y, r, offsets = np.array([0,0])):

    # rectangle to go around the cylinder
    points = np.array(rectangle_(l_x, l_y, offsets))
    segments = np.concatenate((points[1:],points[:3]), axis = 1)
    segments = np.concatenate((segments, np.expand_dims(np.concatenate((points[-1],points[0])),axis = 0)), axis = 0)

    # cylinder itself
    circle = [[c_x+r*np.cos(k*np.pi/res), c_y+r*np.sin(k*np.pi/res), c_x+r*np.cos((k+1)*np.pi/res), c_y+r*np.sin((k+1)*np.pi/res)] for k in range(2*res)]

    return SystemCreator(np.concatenate((segments, circle), axis = 0),  [1,3]), 3
