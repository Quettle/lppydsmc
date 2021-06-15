import numpy as np

def basic(arr, count, law): 
    """ Returns an array containing the indexes of the particles that reacted (and lost) an electron with the walls.

    Args:
        arr (ndarray, float): the array of particles, shape : number of particles x 5 (x,y,vx,vy,vz)
        count (ndarray, int): array of integers containing the number of times the associated particle in *arr* collided
        law (function): returns a probability of reactions ...
    """

    # NOTE : we can sample only for the particles that collided - which would give less computations
    # however it requires a loop (or a copy)
    # or we can draw for the whole array of particles which is crearly suboptimal
    idx_reactions = []
    for k, c in enumerate(count):
        if(c>0):    
            proba_reaction = law(arr[k], c) # TODO we probably have the wall to consider in the future
            rdm_uniform_draw = np.random.random() # we can maybe draw them all out (as we are anyway counting the number of colliding particles outside the function)
            if(proba_reaction < rdm_uniform_draw):
                idx_reactions.append(k)

    return np.array(idx_reactions)