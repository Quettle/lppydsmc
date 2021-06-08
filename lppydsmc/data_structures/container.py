import numpy as np

class Container(object):
    
    def __init__(self, size_array, number_of_elements, dtype):
        self.size_array = size_array
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
    
    # we will have the same issue of not being able to use it I think.
    # @njit # in the long run, we may want to use 
    def delete_multiple(self, idxes):
        # best is Numba - however we can not use sort('stable') (idxes is almost sorted)
        
        # Previous version - copying array takes lots of time for very big arrs
        # we could use np.delete() however it changes the size of the return arrays so we have to be more careful
        # self.arr[:self.size_array-idxes.shape[0],:] = np.delete(self.arr, idxes, axis = 0) # operation is not inplace
        # self.current-=idxes.shape[0]
        
        idxes.sort() # inplace
        # forced to loop 
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