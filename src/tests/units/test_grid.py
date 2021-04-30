import unittest
import numpy as np

from src.utils.grid import Grid
from icecream import ic
class TestGrid(unittest.TestCase):
    rx, ry = 33, 78
    max_nb = 500

    def test_make_grid(self):
        resolutions = np.array((self.rx, self.ry))
        max_number_per_cell = self.max_nb
        return Grid(resolutions = resolutions, max_number_per_cell=max_number_per_cell)
        
    def test_add_and_remove(self):
        grid = self.test_make_grid()

        positions = [np.array([i,j], dtype = int) for i in range(self.rx) for j in range(self.ry)]

        o = np.array([0,0])
        for pos in positions:
            grid.add(pos,o)

        a = grid.get(positions[0])
        current = grid.get_current(positions[0])

        self.assertTrue(np.array_equal(a[0], o))
        self.assertTrue(current==1)

        a = grid.get(positions[-1])
        current = grid.get_current(positions[-1])

        self.assertTrue(np.array_equal(a[0], o))
        self.assertTrue(current==1)

        arr = np.random.randint(low = 0, high = min(self.rx, self.ry), size = (100,4))
        grid.add_multiple(arr)
        
        for row in arr : # row = [pos_x_grid, pos_y_grid, idx_container, idx_in_container]
            pos = row[:2]
            cur = grid.get_current(pos)
            grid.delete(pos, cur-1)

        for pos in positions:
            self.assertTrue(int(grid.get_current(pos))==1, f'Results : {grid.get_current(pos)==1} with current : {grid.get_current(pos)} and pos : {pos}')
            arr = grid.get(pos)[0]
            self.assertTrue(np.array_equal(arr, o))

    # TODO: when adding DYNAMIC array, then I should test it is in fact dynamic


if __name__ == '__main__':
    unittest.main()