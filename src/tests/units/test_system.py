from typing import Type
import unittest
import numpy as np

from src.system_creator import SystemCreator
from icecream import ic

class TestSystem(unittest.TestCase):
    segments = np.array([
        [1,-1,1,0],
        [4,-1,1,-1],
        [2,1,1,0],
        [3,0,2,1],
        [3,0,4,0],
        [4,0,4,-1],
    ], dtype=float)

    def test_make_system(self):
        system = SystemCreator(self.segments)
        walls, a, offsets = system.get_segments(), system.get_dir_vects(), system.get_offsets()
            
        self.assertTrue(np.array_equal(walls[0],np.array([1,-1,1,0], dtype=float)))
        self.assertTrue(np.array_equal(walls[1],np.array([1,-1,4,-1], dtype=float)))
        self.assertTrue(np.array_equal(walls[2],np.array([1,0,2,1], dtype=float)))
        self.assertTrue(np.array_equal(walls[3],np.array([2,1,3,0], dtype=float)))
        self.assertTrue(np.array_equal(walls[4],np.array([3,0,4,0], dtype=float)))
        self.assertTrue(np.array_equal(walls[5],np.array([4,-1,4,0], dtype=float)))

        self.assertTrue(np.array_equal(a[0,:2],np.array([0,1], dtype=float)))
        self.assertTrue(np.array_equal(a[1,:2],np.array([1,0], dtype=float)))
        self.assertTrue(np.array_equal(a[2,:2],1/np.sqrt(2)*np.array([1,1], dtype=float)))
        self.assertTrue(np.array_equal(a[3,:2],1/np.sqrt(2)*np.array([1,-1], dtype=float)))
        self.assertTrue(np.array_equal(a[4,:2],np.array([1,0], dtype=float)))
        self.assertTrue(np.array_equal(a[5,:2],np.array([0,1], dtype=float)))

        self.assertTrue(a[0,2]==1)
        self.assertTrue(a[1,2]==3)
        self.assertTrue(a[2,2]==np.sqrt(2))
        self.assertTrue(a[3,2]==np.sqrt(2))
        self.assertTrue(a[4,2]==1)
        self.assertTrue(a[5,2]==1)
        

        self.assertTrue(offsets[0]==1.)
        self.assertTrue(offsets[1]==-1.)

        extrema = system.get_extremal_values()
        self.assertTrue(extrema['min_x']==1.)
        self.assertTrue(extrema['max_x']==4.)
        self.assertTrue(extrema['min_y']==-1.)
        self.assertTrue(extrema['max_y']==1.)
            
        shape = system.system_shape()
        self.assertTrue(np.array_equal(shape, np.array([3,2], dtype=float)))

    
if __name__ == '__main__':
    unittest.main()