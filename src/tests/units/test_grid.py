import unittest
import numpy as np

from src.utils.grid import Grid

class TestGrid(unittest.TestCase):
    
    def test_make_grid(self):
        grid = Grid((3,2,2))

        N = 12
        list_objects = [k for k in range(N)]
        positions_initialize = [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0),
                                (1,0,1), (1,1,0), (1,1,1), (2,0,0), (2,0,1),
                                (2,1,0), (2,1,1)]
        new_positions = [(2,1,1),(0,0,0), (0,0,1), (0,1,0), (0,1,1), 
                        (1,0,0), (1,0,1), (1,1,0), (1,1,1), (2,0,0), 
                        (2,0,1), (2,1,0)]

        for pos, o in zip(positions_initialize, list_objects):
            grid.add(pos, o)
        
        objects_list = grid.get_all()
        self.assertTrue(all(a==b for a, b in zip(objects_list, [0,1,2,3,4,5,6,7,8,9,10,11])))

        for old_pos, pos, o in zip(positions_initialize, new_positions, list_objects):
            grid.update(old_pos, pos, o)

        objects_list = grid.get_all()
        self.assertTrue(all(a==b for a, b in zip(objects_list, [1,2,3,4,5,6,7,8,9,10,11,0])))

        for pos, o in zip(new_positions, list_objects):
            grid.remove(pos, o)

        objects_list = grid.get_all()
        self.assertTrue(len(objects_list) == 0)

if __name__ == '__main__':
    unittest.main()