import numpy as np

from icecream import ic

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
        self.arr[pos[0], pos[1]][self.current[pos[0], pos[1]]:self.current[pos[0], pos[1]]+o.shape[0]] = o
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

def pos_in_grid(pos, grid_res, offsets, system_shape):
    return np.floor(np.subtract(pos,offsets)*grid_res/system_shape).astype(int)
