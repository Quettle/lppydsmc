# Imports
import numpy as np

# my modules
from src.system_creator import SystemCreator

# -------------------- This Module -------------------------- #
# this module allows an easier use of different default systems 

    # --------------------- Utils functions -------------------- #

def rectangle_(w,l, offset=np.array([0,0])):
    # top left point is p1 and then its anti-trigo rotation
    p1 = np.array([0,l])+offset
    p2 = np.array([w,l])+offset
    p3 = np.array([w,0])+offset
    p4 = np.array([0,0])+offset
    return p1,p2,p3,p4

    # --------------------- Rectangle system ----------------- #

def system_rectangle(lx, ly, offsets = np.array([0,0])):
    # segment order : top, right, bottom, left
    points = np.array(rectangle_(lx, ly, offsets))
    segments = np.concatenate((points[:3], points[1:]), axis = 1)
    segments = np.concatenate((segments, np.expand_dims(np.concatenate((points[-1],points[0])),axis = 0)), axis = 0)
    return SystemCreator(segments), [1,3], 1

    # --------------------- Thruster system ----------------- #

def thruster(w_in, l_in, w1, l1, l_int, w2, l2, w_out, l_out, offsets = np.array([0,0])):
    # hypothesis : w_int = w_in
    # returns an array with the walls for the thruster
    # not optimized but we prioritize the clarity here

    p1, p2, p3, p20 = rectangle_(w_in,l_in, offset = offsets)
    p19, p4, p5, p18 = rectangle_(w1,l1, offset = offsets+np.array([0.5*(w_in-w1),-l_in]))
    p17, p6, p7, p16 = rectangle_(w_in,l_int, offset = offsets+np.array([0, -l1-l_in]))
    p15, p8, p9, p14 = rectangle_(w2, l2, offset = offsets+np.array([0.5*(w_in-w2),-l_in-l1-l_int]))
    p13, p10, p11, p12 = rectangle_(w_out, l_out, offset = offsets+np.array([0.5*(w_in-w_out),-l_in-l1-l_int-l2]))
    points = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20])
    segments = np.concatenate((points[1:],points[:19]), axis = 1)
    segments = np.concatenate((segments, np.expand_dims(np.concatenate((p20,p1)),axis = 0)), axis = 0)
    # sorting is realized when the array is created per the SystemCreator. No need to worry at this point.
    return segments # system, idx_out_walls, idx_in_wall

def thruster_system(w_in, l_in, w1, l1, l_int, w2, l2, w_out, l_out, offsets = np.array([0,0])):
    segments = thruster(w_in, l_in, w1, l1, l_int, w2, l2, w_out, l_out, offsets = offsets)
    return SystemCreator(segments), [0,10], 0 # system, idx_out_walls, idx_in_wall

    # --------------------- Cylinder system ----------------- #

def cylinder_system(res, lx, ly, cx, cy, r, offsets = np.array([0,0])):

    # rectangle to go around the cylinder
    points = np.array(rectangle_(lx, ly, offsets))
    segments = np.concatenate((points[1:],points[:3]), axis = 1)
    segments = np.concatenate((segments, np.expand_dims(np.concatenate((points[-1],points[0])),axis = 0)), axis = 0)

    # cylinder itself
    circle = [[cx+r*np.cos(k*np.pi/res), cy+r*np.sin(k*np.pi/res), cx+r*np.cos((k+1)*np.pi/res), cy+r*np.sin((k+1)*np.pi/res)] for k in range(2*res)]

    return SystemCreator(np.concatenate((segments, circle), axis = 0)), [1,3], 1
