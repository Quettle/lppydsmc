from typing import Type
import unittest
import numpy as np

from src.system_creator import SystemCreator

class TestSystem(unittest.TestCase):
    
    def test_make_system(self):
        pass

    def test_extremal_values(self):
        #system = self.test_make_system()
        #self.assertTrue(all(a==b for a, b in zip(system.get_extremal_values().values(), [0,1,0,2])))
        #self.assertTrue(all(a==b for a,b in zip(system.get_size(), [1,2])))
    
if __name__ == '__main__':
    unittest.main()