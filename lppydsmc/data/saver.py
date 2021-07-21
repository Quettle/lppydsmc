import pandas as pd
import numpy as np
from pathlib import Path
import os
# to save the data during the simulation 
from pprint import pprint

class Saver(object):
    # should be always use with a 'with' statement.
    # that is why there is an __enter__ and __exit__ function

    def __init__(self, dir_path, name, *args):

        # if(os.path.exists(dir_path/name) and not 'append' in args):
        #   os.system('rm -f -r {}'.format(dir_path/name))

        # loading / creating the bucket for the data
        self.dir_path = dir_path
        self.name = name

    def save(self, it, append = None, update = None):
        if(update is not None):
            for k, v in update.items():
                self.store[k] = self._convert(v, it)
        if(append is not None):
            for k, v in append.items():
                try :
                    self.store.append(k,self._convert(v, it))
                except Exception as e :
                    print(k)
                    print(type(v))
                    # pprint(v.dtypes())
                    return e
    # def close(self):
    #   self.store.close()

    def load(self):
        return pd.HDFStore(self.dir_path/self.name)

    def __enter__(self):
        self.store = pd.HDFStore(self.dir_path/self.name)
        return self.store

    def __exit__(self):
        self.store.close()
  

    # ---------------------- Utils ------------------- #    

    def _convert(self, v, it):
        if(type(v) not in [pd.DataFrame, pd.Series]):
            if(type(v) == np.ndarray):
                if(len(v.shape) >= 2):
                    v = v.flatten()
                return pd.Series(v, index = [it]*v.shape[0])
            else:
                # this can yields to errors as we don't know what this is
                # we are supposing we have int or float here
                return pd.Series(v, index = [it])

        else:
            return v