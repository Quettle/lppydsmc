import numpy as np


def background_gas(arr, law): # we need a way to do the reactions with somekind of proba actually close to the "true" probability of reactions.
    # otherwise we can not conclude anything on the results.
    """ Returns an array of indexes, each indexw means the associated particle collided with the background gas.

    Args:
        arr (ndarray 2D): the array of particles, shape : nb of particles x 5
        law (function): function to determine from the particle's position and velocity it's collision probability
    """

    proba = law(arr)

    rdm_uniform_sample = np.random.random(arr.shape[0])

    return np.where(proba > rdm_uniform_sample), np.mean(proba)

