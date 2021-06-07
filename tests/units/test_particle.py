import unittest
import numpy as np
from src.utils import Particle

from icecream import ic

class TestParticle(unittest.TestCase):
    
    def make(self):
        part_type = 'I'
        charge = 1.6e-19 # C
        mass = 126.9*1.7e-27 # kg - roughly
        radius = 2e-10
        size_array = 10000

        container = Particle(part_type, charge, mass, radius, size_array)

        self.assertTrue(np.array_equal(container.get_params(), np.array([mass, charge, radius, 4*np.pi*radius**2])))
        self.assertTrue(container.get_current()==0)
        self.assertTrue(container.get_particles().shape == (0, 5)) # should be empty too
        
        return container

    def add(self):
        container = self.make()
        current = container.get_current()

        p1 = np.array([0.1,-0.02, 233.2203, -103.1028, -0.01], dtype = float)
        container.add(p1)

        self.assertTrue(container.get_current()==current+1)
        self.assertTrue(np.array_equal(container.get_particles()[current], p1)) # should be empty too

        return container

    def add_multiple(self):
        container = self.add()

        parts =  np.array([[-0.0911,-1.10, -23.203, -103.84, -0.01],
                    [-0.1,-2.13, -874.203, -1092173.84, 18383.01]], dtype = float)

        current = container.get_current()

        container.add_multiple(parts)
        
        self.assertTrue(container.get_current()==current+2)
        self.assertTrue(np.array_equal(container.get_particles()[current:container.get_current()], parts)) # should be empty too

        # testing with adding only one particle

        current = container.get_current()

        other_parts = 2*np.array([[0.1,-0.02, 233.2203, -103.1028, -0.01]], dtype = float)
        container.add(other_parts)

        self.assertTrue(container.get_current()==current+1)
        self.assertTrue(np.array_equal(container.get_particles()[current:container.get_current()], other_parts)) # should be empty too

        self.assertTrue(container.get_current()==4)

        return container

    def pop(self):
        container = self.add_multiple()

        current = container.get_current()
        part = container.pop(idx=current-2)

        self.assertTrue(np.array_equal(part, np.array([-0.1,-2.13, -874.203, -1092173.84, 18383.01], dtype = float)))
        self.assertTrue(container.get_current()==current-1)

        return container

    def delete(self):
        container = self.pop()

        current = container.get_current()
        part = container.delete(idx=container.get_current()-1)

        self.assertTrue(part is None) # delete hsould not return anything
        self.assertTrue(container.get_current()==current-1)

        return container

    def test_particle(self):
        container = self.delete()
        self.assertTrue(container.get_current()==2)
        self.assertTrue(np.array_equal(container.get_particles()[:container.get_current()], \
             np.array([[0.1,-0.02, 233.2203, -103.1028, -0.01],[-0.0911,-1.10, -23.203, -103.84, -0.01]], dtype = float)))

if __name__ == '__main__':
    unittest.main()