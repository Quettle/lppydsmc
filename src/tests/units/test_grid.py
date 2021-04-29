import unittest
import numpy as np

from src.utils.grid import Grid

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
            grid.add(np.array(pos, o))

        arr = np.random.uniform(low = 0, high = min(self.rx, self.y), size = (100,4))
        grid.add_multiple(arr)

        for row in arr :
            pos = row[:2]
            grid.delete(pos, 0)

    # TODO: when adding DYNAMIC array, then I should test it is in fact dynamic


if __name__ == '__main__':
    unittest.main()