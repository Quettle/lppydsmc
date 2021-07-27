import numpy as np

def handler_particles_collisions(arr, grid, currents, dt, average, pmax, cross_sections, volume_cell, particles_weight, remains, species_mass = None, monitoring = False, group_fn = None):
    """ Handles the collisions between species in DSMC. It works INPLACE.

    Args:
        arr (list): list of arrays, each is associated with a species and has shape : number of particles in species x 5
        grid (np.ndarray): a 2D-array of (shape number of cells x max size x 2) containing for each cell the particles that are inside.
                           Each particle is describes by its 'container index' (or index in arr, or species index) and its index in the associated species container.
        currents (np.ndarray): 1D-array containing the number of particles in each cell.
        dt (float): time step
        average (np.ndarray): 1D-array containing the average number of particles in each cell
        pmax (np.ndarray): 1D-array of floats of size (number of cells) containing for each cell the normalizing maximum probability.
        cross_sections (np.ndarray): 2D-arrays containing on line i, column j the cross section between species i and j.
        volume_cell (float): volume of a cell (structured mesh)
        particles_weight (int): the particles weight (number of real particles, one simulated one represents)
        remains (np.ndarray): when selecting possible candidates to compute collisions from (using Bird formula), the results is not a integer. 
                              This is the float part that should be passed to the next time step.
        species_mass (np.ndarray, optional): 1D-array containing the mass of the species, useful when particles are not the same mass. Defaults to None.
        monitoring (bool, optional): if it should track the process. In which case is computed : 
                                        - tracking = ['cell_idx','max_proba','mean_proba','mean_number_of_particles','mean_distance'] - shape : nb_cells x 5
                                        - collisions = ['cell_idx','° couples','quantity'] - shape : nb_cells*nb_species**2 x 3
                                    Defaults to False.
        group_fn (function, optional): Function giving the groups for a colliding couples (in order to be saved as such). 
                                       This allows to know what collisions occured (between which species). If monitoring is True, it should not be None.
                                       Defaults to None.
    
    Returns:
        [np.ndarray (optional), np.ndarray (optional)]: arrays describing the collisions process as described in *monitoring*.

    TODO : 
        - it's a little counter intuitive to have average being updated outside this function but pmax inside. Even if there is logical grounds.
    """
    remains[:], cands = candidates(currents, dt, average, pmax, volume_cell, particles_weight, remains)
    nb_cells = grid.shape[0]
    nb_species = len(arr)
    masses = None
    if(monitoring):
        # setting collisions
        nb_groups = (nb_species*(nb_species+1))//2
        n = nb_cells*nb_groups
        collisions = np.zeros((n,3)) # ['cell_idx','° couples','quantity'] # per cell and colliding couples, thus : nb_cells x nb_species**2 lines
        # initializing collisions - we could do it outside ...
        groups_list = np.arange(nb_groups)
        for k in range(nb_cells):
            collisions[k*
            nb_groups:(k+1)*nb_groups] = k
            collisions[k*nb_groups:(k+1)*nb_groups,1] = groups_list

        # setting tracking
        tracking = np.zeros((nb_cells, 5)) # ['cell_idx','max_proba','mean_proba','mean_number_of_particles','mean_distance'], thus nb_cells lines

    count_collisions = 0
    for idx, k in enumerate(currents): # TODO : parallelize # looping over cells right now
        if(cands[idx]>0):
            choice = index_choosen_couples(currents[idx], cands[idx]) # returns couples of colliding particles indexes in GRID - this is why it works !

            g = grid[idx]
            parts = np.array([[g[c[0]], g[c[1]]] for c in choice], dtype = int)
            array = np.array([[ arr[c[0,0]][c[0,1]] , arr[c[1,0]][c[1,1]] ] for c in parts]) # at this point arrays is an array of couples of [idx_container, idx_in_container]

            # selecting the right cross_sections
            if(np.isscalar(cross_sections)):
                cross_sections_couples = cross_sections
            else:
                cross_sections_couples = np.array([cross_sections[c[0,0], c[1,0]] for c in parts])
            
            vr_norm = np.linalg.norm((array[:,1,2:]-array[:,0,2:]), axis = 1)
            d = np.linalg.norm((array[:,1,:2]-array[:,0,:2]), axis = 1)
            
            proba = probability(vr_norm = vr_norm, pmax = pmax[idx], cross_sections = cross_sections_couples)
            
            max_proba = np.max(proba)

            if(max_proba>1):
                pmax[idx] = max_proba*pmax[idx]
            
            colliding_couples = is_colliding(proba) # indexes in array of the ACTUALLY colliding couples

            colliding_array = array[colliding_couples]
            count_collisions+= colliding_couples.shape[0]

            if(monitoring):
                if(colliding_couples.shape[0]>0):
                    # a 2D-array with [[[c1,i1],[c2,i2]], [[c3,i3],[c4,i4]], ...] ; c for container index, i for index in container 
                    # ideally, we would like to do a groupby on container indexes and then count the number 
                    # this is the use of group_fn, defined as a lambda function of 'arr' using first *set_groups* then *get_groups*
                    # and outside this function as it can be setup prior to simulation
                    groups = group_fn(colliding_array.astype(int)) # once we have the groups we can count the quantity for each groups, and save it
                    unique, counts = np.unique(groups, return_counts=True) # so there we may not have all groups in unique
                    collisions[(idx*nb_groups+unique).astype(int),2] = counts

                    # tracking - we could add more than that easily
                    tracking[idx,:] = np.array([idx, pmax[idx], np.mean(proba[colliding_couples]),\
                        average[idx], np.mean(d[colliding_couples])])
                else:
                    tracking[idx,:] = np.array([idx, pmax[idx], np.nan,\
                                            average[idx], np.nan])
            
            # a very long process that should be avoided whenever possible
            if(species_mass is not None):
                colliding_parts = parts[colliding_couples]
                masses = np.zeros((colliding_parts.shape[0],2))
                for k, part in enumerate(colliding_parts):
                    masses[k,0] = species_mass[part[0,0]]
                    masses[k,1] = species_mass[part[1,0]]

            array[colliding_couples] = reflect(colliding_array, vr_norm[colliding_couples], masses)

            for k in range(len(array)):
                c1, c2 = array[k,0], array[k,1]
                c = parts[k]
                arr[c[0,0]][c[0,1]][:] = c1 # copy
                arr[c[1,0]][c[1,1]][:] = c2

    if(monitoring):
        return count_collisions, collisions, tracking # in theory it is useless to return pmax
    return count_collisions
    

def candidates(currents, dt, average, pmax, volume_cell, particles_weight, remains):
    """ Returns the number of candidates couples to perform dsmc collisions between particles. (Note that this formula is for one type of particle only => in fact, it is not)

    Args:
        currents (ndarray - 2D - float): number of particles per cell
        dt (float): time step
        average (ndarray - 2D - float): average number of particle in the cell
        pmax (ndarray - 2D - float): max probability per cell ()
        volume_cell (float or ndarray - 2D - float): volume of a cell
        particles_weight (float): "macro-ratio" - ratio of real particles over macro-particles 

    Returns:
        (ndarray - 2D - float, ndarray - 2D - int) : the number of candidates to select per cell to perform collisions - fractional part first and then int part.
    """
    # for one type of particle for now
    remains, cands = np.modf(0.5*currents*average*pmax*particles_weight/volume_cell*dt+remains) # (Nc Nc_avg particles_weight (sigma vr)max dt)*/(2V_c)
    return remains, cands.astype(int) 

def index_choosen_couples(current, candidates, verbose = False): 
    """ Return as many couple indexes as there is candidates couples of particles that should be processed. The index are in [0, number of particles in current cell].

    Args:
        current (int): number of particles in the current cells
        candidates (int): number of couples to select, which mean 2*candidates particles needs to be selected.
        verbose (bool, optional): If verbose is True, then it will print if there is not enough particles compared to the number that have to be selected.
                                  Note : initially,  *np.random.default_rng().choice* with *replace = False* was used and it returned an error. It was then removed
                                  as the command is not available in all versions of NumPy.
                                  Defaults to False.
        TODO : should the selection of couples be as much as possible without repetitions or is it the right way of doing to randomly select all couples with repetitions ?                 
    Returns:
        [np.ndarray]: 2D-arrays containing the candidates couples (part1, part2) as indexes in the cell.
    """
    return np.random.randint(low = 0, high = current, size = (candidates,2)) 

        # previous version - all NumPy version do not have it
    # try :
    #     return np.random.default_rng().choice(current, size = (candidates, 2), replace=False, axis = 1, shuffle = False)
    # except ValueError as e:
    #     if(verbose):
    #         print(e) 
    #         print(f'{current} < 2 x {candidates} => Some macro-particles will collide several times during the time step.')
    #     return np.random.default_rng().choice(current, size = (candidates, 2), replace=True, axis = 1, shuffle = False) # we dont need shuffling

def probability(vr_norm, pmax, cross_sections): # still per cell
    """ Returns the collisions probability associated with the possibly colliding couples.

    Args:
        vr_norm (np.ndarray): array of size (number of candidates) with the relative speed norm for each couple.
                              Can be computed using something like : np.linalg.norm((arr[choices][:,1,2:]-arr[choices][:,0,2:]), axis = 1)
        pmax (float): maximum probability for the current cell
        cross_sections (np.ndarray): array of size (number of candidates) containing the cross sections for each couple. 
                                     Note that a *float* can also be used and it will return the right result. 
                                     It is useful when dealing with constant cross sections accross couples.

    Returns:
        [np.ndarray]: 1D-array of size (number of candidates) containing the probability of collisions for each couple.
    """
    return cross_sections/pmax*vr_norm

def is_colliding(proba):
    """ Returns the indexes where there is in fact collisions (only the indexes).

    Args:
        proba (np.ndarray): the colliding probability for each possibly colliding couples.

    Returns:
        np.ndarray: returns a 1D-array with the indexes of the actually colliding couples.
    """
    r = np.random.random(size = proba.shape)
    return np.where(proba>r)[0]

def reflect(arr, vr_norm, masses = None):
    """ Reflect particles following their collisions. Reflections are for now purely specular (no randomness).

    Args:
        arr (np.ndarray): 2D-array of size (number of actually colliding couples x 5)
        vr_norm (np.ndarray): relative velocity norm for each colliding couple
        masses (np.ndarray, optional): Mass for each particle, for each couple. If mass is the same for every particle, then use None as it simplified the computations.
                                       Defaults to None.

    Returns:
        np.ndarray  : Returns the new array with velocities having been updated accordingly.
                      Note that *arr* is in fact changed in place, however, we still return it in case it is needed  (in the case of 
                      the main algorithm *handler_particles_collisions*, *arr* is a copy and not a reference to the initial array).
                      E.g. arr = array[is_colliding(proba)]
    """

    if(masses is None):
        coeff1, coeff2 = 0.5, 0.5 # same mass
    else:
        mass_sum = masses[:,0]+masses[:,1]
        coeff1, coeff2 = masses[:,0]/mass_sum, masses[:,1]/mass_sum # this time they are 1D-array of size the number of particles
        coeff1 = np.expand_dims(coeff1, axis = 1)
        coeff2 = np.expand_dims(coeff2, axis = 1)

    r = np.random.random(size = (2,arr.shape[0]))
    ctheta = 2*r[0,:]-1
    stheta = np.sqrt(1-ctheta*ctheta)
    phi = 2*np.pi*r[1,:]
    
    v_cm = coeff1*arr[:,0,2:]+coeff2*arr[:,1,2:]
    v_r_ = np.expand_dims(vr_norm, axis = 1)*np.stack((stheta*np.cos(phi), stheta*np.sin(phi), ctheta),  axis = 1) # 

    arr[:,0,2:] = v_cm + coeff1*v_r_
    arr[:,1,2:] = v_cm - coeff2*v_r_

    return arr

# -------------------------------- Groups functions ------------------------ #

# default groups functions
def set_groups(n):
    """ Associate for each possibile binart collisions an unique index. This allows the tracking.
        The goal is : when monitoring dsmc, for each actually colliding couples we want to save which species collided.
        Another way, instead of saving which species collided, is to save an identifier of the collision (e.g. for 'I' and 'I-', the id is 3).
        The pros are :
            - less savings
            - no difference between (A,B) and (B,A)
            - it is vectorized : if we were to recognize that (A,B) is the same than (B,A) then, it would not be anymore vectorized.
        Of course, it means that when post-processing the data, we have to know which species are associated to the id.
        Note : it's a little bit more complicated for not much improvement because we could have saved simply the species and post-process everything afterwards.
    Args:
        n (int): number of species

    Returns:
        np.ndarray: symmetric matrix with with an identifier for each collision.
    """
    groups = np.zeros((n, n))
    count = 0
    for i in range(n):
        for j in range(i, n):
            groups[i,j] = count
            groups[j,i] = count
            count +=1
    return groups

def get_groups(arr, groups):
    """ Function that returns, the groups associated with the couples given in arr.

    Args:
        arr (np.ndarray): 3D-array of size (candidates x 2 x 2) where a couple is described as : [part1, part2] where part1 = [species_id_1, index_in_array_1], part2 = [species_id_2, index_in_array_2]
        groups (np.ndarray): symmetric matrix with with an identifier for each collision.

    Returns:
        np.ndarray : 1D-array of size (candidates) containing the identifiers for each collision type.
    """
    return groups[arr[:,0,0],arr[:,1,0]]