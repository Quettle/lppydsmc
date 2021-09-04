import numpy as np

def index(array, item):
    """ Find the item in array and returns the index it is at.
    It uses np.array_equal since the arrays store items represented as 1D-array.
    Args:
        array (np.ndarray): 2D-array full of items  
        item (np.ndarray): the item to find in array

    Returns:
        [int]: index of item in array. If not found, None.
    """
    for idx, val  in enumerate(array):
        if np.array_equal(val, item):  # val == item => only for int / float comparisons
            return idx
        
class Grid(object):
    """ Generate a grid to store data in each cell.
    In order to not allocated/deallocate memory during the simulation, each cell bucket is initialized with a max fixed size.

    TODO : add dynamic arrays (meaning we allow increasing the size of the bucket for cells if necessary).
    For now, this is not available. So the *np.empty((size), dtype = np.ndarray)* is : 1) Complicated (array of arrays), 2) Inefficient. 
    It should be replace by : self.arr = np.zeros((size, max_number_per_cell, 2), dtype = int)
    """

    def __init__(self, size, max_number_per_cell, size_entity = 2):
        self.size = size
        self.max_number_per_cell = max_number_per_cell
        self.arr = np.empty((size), dtype = np.ndarray)

        # forced to do that as slicing does not work on arrays of dtype = arrays
        for c in range(self.size):
            self.arr[c]=np.empty((max_number_per_cell, size_entity), dtype = int)
            
        self.current = np.zeros((size), dtype = int)

    def add(self, pos, o): # pos must be a int
        self.arr[pos][self.current[pos]] = o
        self.current[pos]+=1

    def add_multiple(self, positions, objects):
        """ Add objects to the specified positions in grid.

        Args:
            positions (np.ndarray): 1D-array of position in grid of size the number of objects in *objects*.
            objects (np.ndarray): array of shape (number of objects x size_entity) to add to the grid.
        """
        for pos, o in zip(positions, objects):
            self.add(pos, o)

    def _add_multiple(self, pos:int, objects): # works if all data is contiguous in memory and at the same pos
        """ Add multiple objects to a given cell specify by *pos*.

        Args:
            pos (int): the position of the cell objects belong to.
            objects (np.ndarray): array of shape (number of objects x size_entity) to add to grid[pos].

        Raises:
            e: [description]
        """
        try :
            self.arr[pos][self.current[pos]:self.current[pos]+objects.shape[0]] = objects
        except Exception as e:
            print(objects)
            raise e

        self.current[pos] += objects.shape[0]

    def delete(self, pos, idx):
        """Removes the element at index *idx*.
        Args:
            pos (int): position in the grid of the element to be removed
            idx (int): index of the element to be removed
        """
        self.arr[pos][idx] = self.arr[pos][self.current[pos]]
        self.current[pos] -= 1

    def remove(self, pos, o):
        """ Delete the element o at position in grid pos.
        Args:
            pos (int): position in the element in the grid
            o (1d-array of 2 elements): element to be deleted
        """
        idx = index(self.arr[pos][:self.current[pos]], o)
        
        if(idx is None):
            print('Particle not found - not supposed to happen.')
            print(f'pos : {pos}, object :  {o}')

        self.current[pos] -= 1
        self.arr[pos][idx] = self.arr[pos][self.current[pos]]
        
    def update(self, o, old_pos, new_pos):
        self.remove(old_pos, o)
        self.add(new_pos, o)
        
    def update_index(self, pos, idx_container, old_index, new_index):
        # should not be None, if None, there is an issue with the code / simulation at a higher level
        idx = index(self.arr[pos][:self.current[pos]], np.array([idx_container,old_index])) 

        self.arr[pos][idx,1] = new_index # only changing the index in the container 
    
    def reset(self):
        """Reset the indexes of the grids. This is the fastest way for now. It means at each time step, we have to add all the particles again,
        rather than updating their positions, but in Python, this is unfortunately the faster way.
        """
        self.current[::] = 0
    
    # ------------ Getter and setter ------------- #
    def get(self, pos): 
        return self.arr[pos][:self.current[pos]]
    
    def get_current(self, pos):
        return self.current[pos]

    def get_currents(self):
        return self.current
    
    def get_grid(self):
        return self.arr

    def __str__(self) -> str:
        return f'Grid - max size : {self.size}x{self.max_number_per_cell} - filled with {np.sum(self.current)} elements.'

# ---------------- Useful functions to convert from positions in system (2D, float) to position in grid (1D, int) ------------ #

# TODO : not sure this is the right place for these functions.

def default_hashing(positions, res):
    """Convert a number of positions in a 2D-grids to positions in the flattened one.
    Note: this suppose that the 2D grid is flattened row-major.
    
    Args:
        positions (np.ndarray): 2D array of shape (number of items x 2)
        res (int): the resolution along the column

    Returns:
        [np.ndarray]: 1D-array of integers giving the position in the row-major flattened grid
    """
    return positions[:,0]*res+positions[:,1]

def pos_in_grid(pos, grid_res, offsets, system_shape):
    """ Convert positions in the system to positions in a 2D-grid.

    Args:
        pos (np.ndarray): 2D-array of size (number of entities x dimensions)
        grid_res (np.ndarray): resolutions of the grids, size : (dimensions)
        offsets (np.ndarray): offset of the system, when subtracting those from the positions *pos*, the lowest value along each dimension should be 0, size : (dimensions).
        system_shape (np.ndarray): shape of the system, ndarray of float of size (dimensions).

    Returns:
        [np.ndarray]: array of ints size (number of entities x dimensions) containing the positions in the grid.
    """
    return np.floor(np.subtract(pos,offsets)*grid_res/system_shape).astype(int)

def convert_to_grid_format(new, old = 0, container_idx = 0):
    """ Useful specific functions to get the representation in the grid of the particles. 
    Meaning : (container/species index, index in the container), given that te container already has *old* particles saved in it,
    and that we are adding *new-old* new particles to it.

    Args:
        new (int): new number of particles in the container of idx container_idx
        old (int, optional): old number of particles in the container. Defaults to 0.
        container_idx (int, optional): Index of the container. Defaults to 0.

    Returns:
        [np.ndarray]: 2D-array of shape : number of particles added x 2, where each particle is described as : [container_idx, idx_in_container]
    """
    index_container = np.full(shape = (new-old), fill_value = container_idx)
    index_in_container = np.arange(old, new)
    return np.stack((index_container, index_in_container), axis = 1)
    
# def convert_to_grid_datatype(positions, new, old = 0, container_idx = 0):
#     indexes = convert_to_grid_format(new, old = old, container_idx = container_idx)
#     return np.concatenate((positions, indexes), axis = 1).astype(int)