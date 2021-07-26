from .container import Container

import numpy as np

class Particle(Container):
    """ Container for all particles of a given type. Uses the hard sphere model.
    """
    q = 1.6e-19   # C
    me = 9e-31    # electron mass
    mp = 1.7e-27  # proton mass (= neutron mass)
    
    def __init__(self, part_type, charge, mass, radius, size_array):
        """ Initialize an container for all particles of type *part_type*.

        Args:
            part_type (str): the type identifier of the particle e.g. : 'I'
            charge (int): the charge of the particle C
            mass (float): the mass of the particle, kg
            radius (float): the radius of the particle, m
            size_array (int): max size of the array (memory is allocated before the simulation for performance reasons)
        """
        Container.__init__(self, size_array, number_of_elements=5, dtype = float)

        # specific to particles handler
        self.part_type = part_type
        self.params = self._init_params(charge, mass, radius)

    def _init_params(self, charge, mass, radius):
        cross_section = self._compute_cross_section(radius)
        return np.array([mass, charge, radius, cross_section]) # *self.mp, *charge

    def _compute_cross_section(self, radius):
        return np.pi*4*(radius)**2

    # --------------------- Getter and Setter ------------------- #

    def get_params(self):
        return self.params

    def mass(self):
        return self.params[0]
    
    def __str__(self) -> str:
        s = super().__str__()
        part = 'Particle {} : m = {:.3e} kg - q = {:.3e} C, r = {:.3e} m, cs = {:.3e} m2'.format(self.part_type, self.params[0], self.params[1], self.params[2], self.params[3])
        return s + ' - ' + part
