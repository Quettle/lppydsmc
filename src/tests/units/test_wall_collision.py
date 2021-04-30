import unittest

import matplotlib.pyplot as plt
import numpy as np
from src.utils.wall_collision import (handler_wall_collision, make_collisions,
                                      make_collisions_vectorized)

from icecream import ic

class TestSystem(unittest.TestCase):
    # TODO : correct it 
    # TODO : make it all < eps

    segments = np.array([
        [-1,-1,1,-1], # bottom - 0
        [-1,-1,-1,1], # left - 1
        [-1,1,1,1], # top - 2
        [1,-1,1,1] # right - 3
    ], dtype=float) # square 2x2 centered in (0,0).

    a = np.array([
        [1,0,2],
        [0,1,2],
        [1,0,2],
        [0,1,2] # last is the norm of the wall it directs
    ], dtype = float)
    
    def plot_system(self, segments):
        for segment in segments :
            plt.plot([segment[0], segment[2]], [segment[1], segment[3]], color = 'k')
            
    def plot_part(self, p1, p2):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color = 'b')
        plt.arrow(x = p1[0], y = p1[1], dx = p1[2]*0.1, dy = p1[3]*0.1, color = 'r')
        plt.arrow(x = p2[0], y = p2[1], dx = p2[2]*0.1, dy = p2[3]*0.1, color = 'r')

    def test_wall_collision(self):
        visual_debug = False

        segments = self.segments
        a = self.a
        
        radius = 0.1

        eps = 1e-3

        # particles
        arr1 = np.array([2,0,1,0,0],dtype = float)
        arr2 = np.array([-2,0,-0.5,0,0],dtype = float)
        arr3 = np.array([0,2,0,1,0],dtype = float)
        arr4 = np.array([0,-2,0,-1,0],dtype = float)
        arr5 = np.array([0,0,0,0,1],dtype = float)
        arr6 = np.array([2,2,1,1,0], dtype = float)
        arr7 = np.array([1,2,0.5,1,0], dtype = float)
        arr = np.stack((arr1,arr2,arr3,arr4,arr5,arr6,arr7), axis = 0)

        ct, cp = handler_wall_collision(arr, segments, a, radius)
        idxes = np.argmin(ct, axis = 1)

            # Checking collision time, position and wall index
        # arr1 - collision with right wall (3)
        self.assertTrue(idxes[0]==3)  
        self.assertTrue(np.abs(ct[0,idxes[0]] - 1.1) < eps) 
        self.assertTrue(np.linalg.norm(cp[0,idxes[0]] - np.array([0.9,0]))<eps)

        # arr2 - collision with left wall (1)
        self.assertTrue(idxes[1]==1)
        self.assertTrue(np.abs(ct[1,idxes[1]] - 2.2) < eps) 
        self.assertTrue(np.linalg.norm(cp[1,idxes[1]] - np.array([-0.9,0]))<eps)

        # arr3 - collision with top wall (2)
        self.assertTrue(idxes[2]==2)
        self.assertTrue(np.abs(ct[2,idxes[2]] - 1.1) < eps) 
        self.assertTrue(np.linalg.norm(cp[2,idxes[2]] - np.array([0,0.9]))<eps)

        # arr4 - collision with bottom wall (0)
        self.assertTrue(idxes[3]==0) 
        self.assertTrue(np.abs(ct[3, idxes[3]] - 1.1) < eps) 
        self.assertTrue(np.linalg.norm(cp[3, idxes[3]] - np.array([0,-0.9]))<eps)

        # arr5 - collision with no wall
        self.assertTrue(idxes[4]==0) # will take the first wall 
        self.assertTrue(np.isinf(ct[4,idxes[4]])) 
        self.assertTrue(all([np.isnan(cp[4,idxes[4]][k]) for k in range(2)]))

        # arr6 - collision at the intersection between top and right - should choose the first wall
        # that comes in the segments array which means top.
        self.assertTrue(idxes[5]==2) 
        self.assertTrue(np.abs(ct[5, idxes[5]] - 1.1) < eps) 
        self.assertTrue(np.linalg.norm(cp[5, idxes[5]] - np.array([0.9,0.9]))<eps)

        # arr7 - top wall
        self.assertTrue(idxes[6]==2) 
        self.assertTrue(np.abs(ct[6, idxes[6]] - 1.1) < eps) 
        self.assertTrue(np.linalg.norm(cp[6, idxes[6]] - np.array([0.45,0.9]))<eps)

            # Testing now reflectioon
                # vectorized algo
        arr_reflected_vect = make_collisions_vectorized(np.copy(arr), a, ct, cp)
            # p1
        new_pos, new_v = arr_reflected_vect[0,:2], arr_reflected_vect[0,2:]
        self.assertTrue(np.linalg.norm(new_pos-np.array([-0.2,0], dtype = float)) < eps)
        self.assertTrue(np.linalg.norm(new_v+arr1[2:]) < eps)

        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(arr1, arr_reflected_vect[0])
            self.plot_system(segments)
            plt.show()

            # p2
        new_pos, new_v = arr_reflected_vect[1,:2], arr_reflected_vect[1,2:]
        self.assertTrue(np.linalg.norm(new_pos-np.array([0.2,0])) < eps)
        self.assertTrue(np.linalg.norm(new_v+arr2[2:]) < eps)
    
        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(arr2, arr_reflected_vect[1])
            self.plot_system(segments)
            plt.show()

            # p3
        new_pos, new_v = arr_reflected_vect[2,:2], arr_reflected_vect[2,2:]
        self.assertTrue(np.linalg.norm(new_pos-np.array([0,-0.2])) < eps)
        self.assertTrue(np.linalg.norm(new_v+arr3[2:]) < eps)
        
        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(arr3, arr_reflected_vect[2])
            self.plot_system(segments)
            plt.show()

            # p4
        new_pos, new_v = arr_reflected_vect[3,:2], arr_reflected_vect[3,2:]
        self.assertTrue(np.linalg.norm(new_pos-np.array([0,0.2])) < eps)
        self.assertTrue(np.linalg.norm(new_v+arr4[2:]) < eps)

        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(arr4, arr_reflected_vect[3])
            self.plot_system(segments)
            plt.show()
        
            # p5
            # p6
        new_pos, new_v = arr_reflected_vect[5,:2], arr_reflected_vect[5,2:]
        self.assertTrue(np.linalg.norm(new_pos-np.array([2,-0.2])) < eps)
        self.assertTrue(np.linalg.norm(new_v-np.array([arr6[2], -arr6[3], arr6[4]], dtype = float)) < eps)
        
        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(arr6, arr_reflected_vect[5])
            self.plot_system(segments)
            plt.show()

           # p7
        new_pos, new_v = arr_reflected_vect[6,:2], arr_reflected_vect[6,2:]
        self.assertTrue(np.linalg.norm(new_pos-np.array([1,-0.2])) < eps)
        self.assertTrue(np.linalg.norm(new_v-np.array([arr7[2], -arr7[3], arr7[4]], dtype = float)) < eps)

        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(arr7, arr_reflected_vect[6])
            self.plot_system(segments)
            plt.show()


                    # loop algo
        arr_reflected = make_collisions(np.copy(arr), a, ct, cp)
            # p1
        new_pos, new_v = arr_reflected[0,:2], arr_reflected[0,2:]
        self.assertTrue(np.linalg.norm(new_pos-np.array([-0.2,0], dtype = float)) < eps)
        self.assertTrue(np.linalg.norm(new_v+arr1[2:]) < eps)

        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(arr1, arr_reflected[0])
            self.plot_system(segments)
            plt.show()

            # p2
        new_pos, new_v = arr_reflected[1,:2], arr_reflected[1,2:]
        self.assertTrue(np.linalg.norm(new_pos-np.array([0.2,0])) < eps)
        self.assertTrue(np.linalg.norm(new_v+arr2[2:]) < eps)
    
        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(arr2, arr_reflected[1])
            self.plot_system(segments)
            plt.show()

            # p3
        new_pos, new_v = arr_reflected[2,:2], arr_reflected[2,2:]
        self.assertTrue(np.linalg.norm(new_pos-np.array([0,-0.2])) < eps)
        self.assertTrue(np.linalg.norm(new_v+arr3[2:]) < eps)
        
        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(arr3, arr_reflected[2])
            self.plot_system(segments)
            plt.show()

            # p4
        new_pos, new_v = arr_reflected[3,:2], arr_reflected[3,2:]
        self.assertTrue(np.linalg.norm(new_pos-np.array([0,0.2])) < eps)
        self.assertTrue(np.linalg.norm(new_v+arr4[2:]) < eps)

        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(arr4, arr_reflected[3])
            self.plot_system(segments)
            plt.show()
        
            # p5
            # p6
        new_pos, new_v = arr_reflected[5,:2], arr_reflected[5,2:]
        self.assertTrue(np.linalg.norm(new_pos-np.array([2,-0.2])) < eps)
        self.assertTrue(np.linalg.norm(new_v-np.array([arr6[2], -arr6[3], arr6[4]], dtype = float)) < eps)
        
        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(arr6, arr_reflected[5])
            self.plot_system(segments)
            plt.show()

           # p7
        new_pos, new_v = arr_reflected[6,:2], arr_reflected[6,2:]
        self.assertTrue(np.linalg.norm(new_pos-np.array([1,-0.2])) < eps)
        self.assertTrue(np.linalg.norm(new_v-np.array([arr7[2], -arr7[3], arr7[4]], dtype = float)) < eps)

        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(arr7, arr_reflected[6])
            self.plot_system(segments)
            plt.show()


if __name__ == '__main__':
    unittest.main()
