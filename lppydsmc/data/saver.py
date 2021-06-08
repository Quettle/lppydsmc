import pandas as pd
import numpy as np
from pathlib import Path
import os
# to save the data during the simulation 

class Saver(object):

    def __init__(self, dir_path, name, *args):

        if(os.path.exists(dir_path/name) and not 'append' in args):
            os.system('rm -f -r {}'.format(dir_path/name))

        # loading / creating the bucket for the data
        self.store = pd.HDFStore(dir_path/name)
       
        self.dir_path = dir_path
        self.name = name

    def save(self, it, append = None, update = None):
        if(update is not None):
            for k, v in update.items():
                self.store[k] = self._convert(v, it)
        if(append is not None):
            for k, v in append.items():
                self.store.append(k,self._convert(v, it))

    def close(self):
        self.store.close()

    def load(self):
        return pd.HDFStore(self.dir_path/self.name)

    # ---------------------- Utils ------------------- #

    def _convert(self, v, it):
        if(type(v) != pd.DataFrame and type(v) != pd.Series):
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