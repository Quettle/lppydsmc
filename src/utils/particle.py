import numpy as np

class Particle(object):
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
        self.size_array = size_array
        self.arr = np.empty(shape = (size_array, 5), dtype = float)
        self.current = 0
        self.part_type = part_type

        self.params = self._init_params(charge, mass, radius)

    def _init_params(self, charge, mass, radius):
        cross_section = self._compute_cross_section(radius)
        return np.array([mass, charge, radius, cross_section]) # *self.mp, *charge

    def _compute_cross_section(self, radius):
        return np.pi*4*(radius)**2

    # -------------------- Updating the list -------------------- #
    def add(self, o): # pos must be a tuple
        self.arr[self.current] = o
        self.current+=1

    def add_multiple(self, o):
        self.arr[self.current:self.current+o.shape[0]] = o[:]
        self.current+=o.shape[0]
    
    def delete_multiple(self, idxes):
        # we could use np.delete() however it changes the size of the return arrays so we have to be more careful
        self.arr[:self.size_array-idxes.shape[0],:] = np.delete(self.arr, idxes, axis = 0) # operation is not inplace
        self.current-=idxes.shape[0]

    def delete(self, idx):
        """Removes the element at index *idx*.

        Args:
            idx (int): index of the element to be removed
        """
        self.arr[idx] = self.arr[self.current-1]
        self.current -= 1
    
    def pop(self, idx):
        """Removes the element at index *idx* and returns it.

        Args:
            idx (int): index of the element to be removed

        Returns:
            ndarray: the removed element
        """
        tmp = np.copy(self.arr[idx])
        self.arr[idx] = self.arr[self.current-1] # the last one is moved before
        self.current -= 1
        return tmp

    def remove(self, o):
        """Removes the first element of the list corresponding to o (by reference, not value).

        Args:
            o (ndarray): the array to be removed.
        """
        for idx in range(self.current):
            if(self.arr[idx] == o): # same object # np.array_equal(self.arr[idx], o) => same values
                self.arr[idx] = self.arr[self.current-1]
                self.current -= 1
                break

    # --------------------- Getter and Setter ------------------- #

    def get_params(self):
        return self.params
    
    def get_current(self):
        return self.current

    def get_particles(self):
        return self.arr[:self.current]