from typing import Type
import unittest
import numpy as np

from src.utils.vector import Vector
from src.utils.segment import Segment

class TestSegment(unittest.TestCase):

    def test_1(self):
        s = Segment(Vector(0,0), Vector(1,0))
        self.assertEqual(s.get_n(), Vector(0,1), f"Should be {Vector(0,1)}")

    def test_2(self):
        s = Segment([0,0],[1,0])
        self.assertEqual(s.get_n(), Vector(0,1), f"Should be {Vector(0,1)}")

    def test_3(self):
        s = Segment((0,0),(1,0))
        self.assertEqual(s.get_n(), Vector(0,1), f"Should be {Vector(0,1)}")

    def test_4(self):
        s = Segment(Vector(0,0), Vector(5,0))
        self.assertEqual(s.get_n(), Vector(0,1), f"Should be {Vector(0,1)}")

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            Segment(np.array([0,0]),np.array([1,0]))
        
    def test_dimension_error(self):
        with self.assertRaises(AssertionError):
            Segment([0],[1,0])

    def test_equality(self):
        s1 = Segment(Vector(1,2), Vector(5,0))
        s2 = Segment(Vector(1,2), Vector(5,0))
        s3 = Segment(Vector(5,0), Vector(1,2))
        self.assertEqual(s1, s2, f"{s1} should be equal to {s2}.")
        self.assertFalse(s1==s3, f"{s1} should not be equal to {s3}")
    
if __name__ == '__main__':
    unittest.main()