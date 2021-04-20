from typing import Type
import unittest
import numpy as np

from src.utils.vector import Vector
from src.utils.segment import Segment
from src.system_creator import SystemCreator

class TestSystem(unittest.TestCase):
    
    def test_make_system(self):
        p1 = Vector(0,0)
        p2 = Vector(1,0)
        p3 = Vector(1,1)
        p4 = Vector(0.75,2)
        p5 = Vector(0.5,1)
        p6 = Vector(0.25,2)
        p7 = Vector(0,1)

        system = SystemCreator([
            Segment(p1,p2),
            Segment(p2,p3),
            Segment(p3,p4),
            Segment(p4,p5),
            Segment(p5,p6),
            Segment(p6,p7),
            Segment(p7,p1)
        ])

        return system

    def test_extremal_values(self):
        system = self.test_make_system()
        self.assertTrue(all(a==b for a, b in zip(system.get_extremal_values().values(), [0,1,0,2])))
        self.assertTrue(all(a==b for a,b in zip(system.get_size(), [1,2])))
    
if __name__ == '__main__':
    unittest.main()