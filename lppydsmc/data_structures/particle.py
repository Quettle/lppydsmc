from .container import Container

import numpy as np

class Particle(Container):
    """Container for all particles of a given type. Uses the hard sphere model. 

    Return a container with method [...].
    """
    q = 1.6e-19   # C
    me = 9e-31    # electron mass
    mp = 1.7e-27  # proton mass (= neutron mass)
    
    def __init__(self, part_type, charge, mass, radius, size_array):
        """ Initialize an container for all particles of type *part_type*.

        Args:
            part_type (str): the type of the particle e.g. : 'I'
            charge (int): the charge of the particle (then multiplied by the elementary charge) => Nope
            mass (float): the atomic mass of the particle => Nope
            radius (float): the radius of the particle
            size_array (int): max size of the array (memory is allocated before the simulation for performance arguments)
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