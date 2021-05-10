import numpy as np

def thruster(w_in, l_in, w1, l1, l_int, w2, l2, w_out, l_out, offsets = np.array([0,0])):
    # hypothesis : w_int = w_in
    # returns an array with the walls for the thruster
    # not optimized but we prioritize the clarity here
    def rectangle(w,l, offset=np.array([0,0])):
        # top left point is p1 and then its trigo rotation
        p1 = np.array([0,0])+offset
        p2 = np.array([w,0])+offset
        p3 = np.array([w,-l])+offset
        p4 = np.array([0,-l])+offset
        return p1,p2,p3,p4

    p1, p2, p3, p20 = rectangle(w_in,l_in)
    p19, p4, p5, p18 = rectangle(w1,l1, offset = np.array([0.5*(w_in-w1),-l_in]))
    p17, p6, p7, p16 = rectangle(w_in,l_int, offset = np.array([0, -l1-l_in]))
    p15, p8, p9, p14 = rectangle(w2, l2, offset = np.array([0.5*(w_in-w2),-l_in-l1-l_int]))
    p13, p10, p11, p12 = rectangle(w_out, l_out, offset = np.array([0.5*(w_in-w_out),-l_in-l1-l_int-l2]))
    points = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20])
    segments = np.concatenate((points[1:],points[:19]), axis = 1)
    segments = np.concatenate((segments, np.expand_dims(np.concatenate((p20,p1)),axis = 0)), axis = 0)
    # sorting is realized when the array is created per the SystemCreator. No need to worry at this point.
    return segments
