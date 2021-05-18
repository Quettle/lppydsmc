import numpy as np


class Grid(object):
    # Note : this 2D grid is not efficient as it is a grid of object of type ndarray and 2D.
    # What could be done is to make a big 4D grid
    def __init__(self, resolutions, max_number_per_cell):
        self.resolutions = resolutions
        self.arr = np.empty(resolutions, dtype = np.ndarray) # not sure it really works actually

        # forced to do that as slicing does not work on arrays of dtype = arrays
        for lx in range(self.resolutions[0]):
            for ly in range(self.resolutions[1]):
                self.arr[lx,ly]=np.empty((max_number_per_cell, 2), dtype = int)
            
        self.current = np.zeros(resolutions, dtype = int)

    def add(self, pos, o): # pos must be a tuple
        self.arr[pos[0], pos[1]][self.current[pos[0], pos[1]]] = o
        self.current[pos[0], pos[1]]+=1

    def add_multiple(self, new_arr):
        # new_arr[k] is for one partice and is : [pos_x, pos_y, idx_container, idx_particle_in_container]
        new_arr = np.sort(new_arr.view('i8,i8,i8,i8'), order = ['f0','f1'], axis = 0).view(int)
        pos_in_grids, indexes = np.unique(new_arr, return_index = True, axis = 0)
        pos_in_grids = pos_in_grids[:,:2].astype(int)
        l = len(pos_in_grids)
        for k in range(1, l):
            pos = pos_in_grids[k-1]
            o = new_arr[indexes[k-1]:indexes[k], 2:]
            self._add_multiple(pos, o)
        if(l>1):
            pos = pos_in_grids[l-1]
            o = new_arr[indexes[l-1]:, 2:]
            self._add_multiple(pos, o)

    def _add_multiple(self, pos, o):
        try :
            self.arr[pos[0], pos[1]][self.current[pos[0], pos[1]]:self.current[pos[0], pos[1]]+o.shape[0]] = o
        except Exception as e:
            ic(o)
            raise e
        # try :
        # except Exception as e: 
            #ic('Error :')
            #ic(e)
            #ic(pos)
            #ic(o)
        # try :        
        #     self.arr[pos[0], pos[1]][self.current[pos[0], pos[1]]:self.current[pos[0], pos[1]]+o.shape[0]] = o
        # except Exception as e:
        #     ic(e)        
        #     ic(f'Max : {len(self.arr[pos[0], pos[1]])}')
        #     ic(f' pos : \n {pos} \n current : \n {self.current[pos[0], pos[1]]} \n shape o : \n {o.shape[0]}')
        #     ic(self.arr[pos[0], pos[1]][self.current[pos[0], pos[1]]:self.current[pos[0], pos[1]]+o.shape[0]])
        #     raise e

        self.current[pos[0], pos[1]] += o.shape[0]

    def delete(self, pos, idx):
        """Removes the element at index *idx*.

        Args:
            pos (tuple): position in the grid of the element to be removed
            idx (int): index of the element to be removed
        """
        self.arr[pos[0], pos[1]][idx] = self.arr[pos[0], pos[1]][self.current[pos[0], pos[1]]]
        self.current[pos[0], pos[1]] -= 1

    def reset(self):
        """Reset the indexes of the grids.
        """
        self.current[:,:] = 0
        
    # ------------ Getter and setter ------------- #
    def get(self, pos): 
        return self.arr[pos[0], pos[1]][:self.current[pos[0], pos[1]]]
    
    def get_current(self, pos):
        return self.current[pos[0], pos[1]]

    def get_currents(self):
        return self.current
    
    def get_grid(self):
        return self.arr


def pos_in_grid(pos, grid_res, offsets, system_shape):
    return np.floor(np.subtract(pos,offsets)*grid_res/system_shape).astype(int)

def convert_to_grid_datatype(positions, new, old = 0):
    index_container = np.zeros((new-old))
    index_in_container = np.arange(old, new)
    indexes = np.stack((index_container, index_in_container), axis = 1)
    return np.concatenate((positions, indexes), axis = 1).astype(int)


# def position_in_grid(pos, system_size, grid_size):
#     """ This functions returns the simples version of 'how to compute a given position in a grid when given the size of the system and the size of the grid'.
#        /!\ It supposes that the particle is in the system. /!\
#         For example : *int(-0.5)* is equal to  *0* thus it will consider such a position to be in the grid whereas it is obviously not. 


#     Args:
#         pos (tuple or list of float): the position to be 'converted'
#         system_size (tuple or list of float): the nd-size of the system
#         grid_size (tuple or list of int): the nd-size of the grid (should be int)

#     Returns:
#         pos_in_grid: a tuple forming the position in grid of the given input position.
#     """
#     pos_in_grid = tuple([int(a*c/b) for a,b,c in zip(pos, system_size, grid_size)])
#     return pos_in_grid
