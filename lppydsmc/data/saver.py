import pandas as pd
import numpy as np
from pathlib import Path
import os

class Saver(object):
    """ Saver object used to save the data to a hdf5 file during the simulation.
    It should always be used with a 'with' statement (that is why there are the __enter__ and __exit__ functions).
    In addition, once created simply using the directory path and name of the file, you can update or add files simply by
    calling the *save* functions with *append* and *update* being two dicts, preferably containing 2D - DataFrames / DataSeries for values.
    Generally speaking, it is also better to have only floats / integers (no string) in the types. It somewhat create issues when saving / adding.
    """

    def __init__(self, dir_path, name, *args):

        # by default will remove the previous file instead of adding to it, unless you have 'append' in the args.
        if(os.path.exists(dir_path/name) and not 'append' in args):
           os.system('rm -f -r {}'.format(dir_path/name))

        # loading / creating the bucket for the data
        self.dir_path = dir_path
        self.name = name

    def save(self, it, append = None, update = None):
        if(update is not None):
            for k, v in update.items():
                self.store[k] = self._convert(v, it)
        if(append is not None):
            for k, v in append.items():
                self.store.append(k,self._convert(v, it))
   
    def load(self):
        return pd.HDFStore(self.dir_path/self.name)

    def __enter__(self):
        self.store = pd.HDFStore(self.dir_path/self.name)
        return self.store

    def __exit__(self):
        self.store.close()
  

    # ---------------------- Utils ------------------- #    

    def _convert(self, v, it):
        """ Default convert functions. The hdf5 format only accepts DataFrames or Series, 
        thus we try to convert what not already of such types to it. It is better than nothing 
        but you should definitely not rely on it. The best way is simply to send simple 2D DataFrames or Series directly.

        Args:
            v (object): an object of unknown type that is to be converted to pandas Series
            it ([type]): [description]

        Returns:
            [pd.Series]: [description]
        """
        if(type(v) not in [pd.DataFrame, pd.Series]):
            if(type(v) == np.ndarray):
                if(len(v.shape) >= 2): # flattening everything that is higher than 2D as we would not know how to store it. 
                    v = v.flatten()
                return pd.Series(v, index = [it]*v.shape[0])
            else:
                # this can yields to errors as we don't know what this is
                # we are supposing we have int or float here
                return pd.Series(v, index = [it])
        else:
            return v