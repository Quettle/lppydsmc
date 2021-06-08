import numpy as np

def index(array, item):
    for idx, val  in enumerate(array):
        if np.array_equal(val, item):  # val == item:
            return idx
        
class Grid(object):
    # this will work with a hashing function

    def __init__(self, size, max_number_per_cell):
        self.size = size
        self.arr = np.empty((size), dtype = np.ndarray) # not sure it really works actually

        # forced to do that as slicing does not work on arrays of dtype = arrays
        for c in range(self.size):
                self.arr[c]=np.empty((max_number_per_cell, 2), dtype = int)
            
        self.current = np.zeros((size), dtype = int)

    def add(self, pos, o): # pos must be a int
        self.arr[pos][self.current[pos]] = o
        self.current[pos]+=1

    # Simplifying it
    def add_multiple(self, positions, objects):
        for pos, o in zip(positions, objects):
            self.add(pos, o)

    def _add_multiple(self, pos, o):
        try :
            self.arr[pos][self.current[pos]:self.current[pos]+o.shape[0]] = o
        except Exception as e:
            print(o)
            raise e

        self.current[pos] += o.shape[0]

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
        
        if(idx is None): # element not found
            print('Particle not found - not supposed to happen.')
            print(f'pos : {pos}, object :  {o}')

            # return False
        # then the element was found
        self.current[pos] -= 1
        self.arr[pos][idx] = self.arr[pos][self.current[pos]]
        # return True

    def update(self, o, old_pos, new_pos):
        # if(old_pos == new_pos): # issue is, o may have changed nonetheless - like the new position of the particle in the array has changed => which is not suppose to happen though
        # thus we may not find it...
        #    return True
        # b = 
        self.remove(old_pos, o)
        # if(b):
        self.add(new_pos, o)
        # return b
    
    def update_index(self, pos, idx_container, old_index, new_index):
        idx = index(self.arr[pos][:self.current[pos]], np.array([idx_container,old_index])) # index in the array of the object we are looking for # should not be None
        self.arr[pos][idx,1] = new_index # only changing the index in the container 
    
    def reset(self):
        """Reset the indexes of the grids.
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

# we now need a hashing function to map from [i,j] (which we get with the pos_in_grid to a only int in [0,res_x*res_y]

def default_hashing(positions, res_y):
    # returns an 1D array of int *
    return positions[:,0]*res_y+positions[:,1]

def pos_in_grid(pos, grid_res, offsets, system_shape):
    return np.floor(np.subtract(pos,offsets)*grid_res/system_shape).astype(int)

def convert_to_grid_format(new, old = 0, container_idx = 0):
    index_container = np.full(shape = (new-old), fill_value = container_idx)
    index_in_container = np.arange(old, new)
    return np.stack((index_container, index_in_container), axis = 1)
    
def convert_to_grid_datatype(positions, new, old = 0, container_idx = 0):
    indexes = convert_to_grid_format(new, old = old, container_idx = container_idx)
    return np.concatenate((positions, indexes), axis = 1).astype(int)