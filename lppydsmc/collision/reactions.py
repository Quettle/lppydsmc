import numpy as np


def background_gas(arr, law):
    """ Returns an array of indexes, each index means the associated particle collided with the background gas.

    Args:
        arr (ndarray 2D): the array of particles, shape : nb of particles x 5
        law (function): function to determine from the particle's position and velocity it's collision probability
    """

    proba = law(arr)

    rdm_uniform_sample = np.random.random(arr.shape[0])

    return np.where(proba > rdm_uniform_sample)

