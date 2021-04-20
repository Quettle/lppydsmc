from numpy.lib.index_tricks import ndenumerate
from src.utils.linkedlist import LinkedList
from src.utils.vector import Vector
import numpy as np


class Grid(object):

    debug = False

    def __init__(self, resolutions):
        self.resolutions = resolutions
        self.data_structure_class = LinkedList
        self.grid = np.empty(resolutions, dtype = self.data_structure_class)
        
        # TODO: improved grid, which can, for example, save the number of objets / cell etc.
        self.number_of_objects = 0

    def add(self, pos, o): # pos must be a tuple
        self.add_(pos, o)
        self.number_of_objects += 1

    def add_(self, pos, o): 
        if(self.grid[pos] == None): # checking if we already created this list or not
            self.grid[pos] = self.data_structure_class()
            
        self.grid[pos].insert(o)

    def remove(self, pos, o):
        self.remove_(pos, o)
        self.number_of_objects -= 1

    def remove_(self, pos, o):
        self.grid[pos].delete(o)

    def update(self, old_pos, pos, o):

        if(old_pos != pos):    
            try :
                self.add_(pos, o)
            except IndexError: # (ValueError,):
                print("New position {} not in grid.".format(pos))
                return False
            self.remove_(old_pos, o)
        else :
            if(self.debug):
                print("Same positions.", end = "   [OK]")

        if(self.debug):
            print("     [OK]")
        return True
    
    # ------------ Getter and setter ------------- #
    def get(self, pos, return_list = True): 
        data_structure = self.grid[pos]
        if(return_list):
            return data_structure.to_list()
        else :
            return data_structure
    
    def get_all(self):
        objects = []
        for idxes, data_structure in np.ndenumerate(self.grid):
            if(data_structure!=None):
                for o in data_structure.to_list():
                    objects.append(o)
        return objects

    def get_grid(self):
        return self.grid
    