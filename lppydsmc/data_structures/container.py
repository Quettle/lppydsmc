import numpy as np

class Container(object):
    """ Describes a simple container for entities that can be described by *number_of_elements* elements.
    The idea is that once created, the Container does not allocate / deallocated any memory. 
    It is created as an array of max size *size_array* and a pointer (self.current) gives the current position.
    It also gives a certain number of useful functions on top of it.
    """
    def __init__(self, size_array:int, number_of_elements:int, dtype):
        self.size_array = size_array
        self.number_of_elements = number_of_elements
        if(number_of_elements != 0):
            self.arr = np.empty(shape = (size_array, number_of_elements), dtype = dtype)
        else :
            self.arr = np.empty(shape = (size_array), dtype = dtype)
        self.current = 0
    
    # -------------------- Updating the list -------------------- #
    def add(self, o): # pos must be a tuple
        self.arr[self.current] = o
        self.current+=1

    def add_multiple(self, o):
        self.arr[self.current:self.current+o.shape[0]] = o[:]
        self.current+=o.shape[0]
    
    # @njit # in the long run, we may want to use that, however the whole class needs to use it.
    # best is Numba - however we can not use sort('stable') (idxes is almost sorted)
    def delete_multiple(self, idxes, sort = True):
        """Delete multiple entities in the smartest way possible in order to diminish computations as much as possible.
            - iterating from the end to conserve indexes
            - simply swapping entities to be deleted with the last active one in the array
        Args:
            idxes (np.ndarray or list): the indexes of the entities to remove.
            sort (bool, optional): if *idxes* are already sorted. Defaults to True.
        """
        if(sort): idxes.sort() # inplace
        # forced to loop but it's better than using np.remove which is not inplace.
        for idx in np.flip(idxes): # view = constant time 
            self.arr[idx] = self.arr[self.current-1]
            self.current -= 1

    def delete(self, idx):
        """Removes the element at index *idx*.

        Args:
            idx (int): index of the element to be removed
        """
        self.arr[idx] = self.arr[self.current-1]
        self.current -= 1

    def pop_multiple(self, idxes, sort = True):
        """ Same as *delete_multiple* but returns it.

        Args:
            idxes (np.ndarray or list): the indexes of the entities to remove.
            sort (bool, optional): if *idxes* are already sorted. Defaults to True.

        Returns:
            [np.ndarray]: entities that were deleted
        """
        if(sort): idxes.sort() # inplace
        tmp = self.arr[idxes] # np.copy(self.arr[idxes]) # np.copy is useless here as self.arr[idxes] is already a copy.
        for idx in np.flip(idxes): # view = constant time
            self.arr[idx] = self.arr[self.current-1]
            self.current -= 1
        return tmp
        
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
           References must be the same (same object).
        Args:
            o (ndarray): the array to be removed.
        """
        for idx in range(self.current):
            if(self.arr[idx] == o): # same object # np.array_equal(self.arr[idx], o) => same values
                self.arr[idx] = self.arr[self.current-1]
                self.current -= 1
                break
    
    def update(self, idx, val):
        self.arr[idx] = val
    # --------------------- Getter and Setter ------------------- #
    
    def get_current(self):
        return self.current

    def get_array(self):
        return self.arr[:self.current]
    
    def get(self, idx):
        return self.arr[idx]

    def get_max_size(self):
        return self.size_array

    def __str__(self) -> str:
        return f'Container filled at {self.number_of_elements} x {self.current}/{self.size_array}'