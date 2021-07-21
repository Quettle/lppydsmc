import numpy as np
# TODO :
# * Test it for one type of particle
# * See formulas for various types of particles (mass, cross section) - it would need an adaptations as we have to store (pmax)pq for each p, q species. 
# It makes it complicated.
# Also note that there is a huge overhead calling python functions etc.
# so I should not DO everything like that but maybe include it in a bigger functions
# which would be much better

def handler_particles_collisions(arr, grid, currents, dt, average, pmax, cross_sections, volume_cell, particles_weight, remains, monitoring = False, group_fn = None):
    # group_fn should not be None if monitoring is True
    # arr : list of arrays
    # works in place for arr but may take very long ...
    # TODO : may return acceptance rates, and stuff like that...
    remains[:], cands = candidates(currents, dt, average, pmax, volume_cell, particles_weight, remains)
    nb_cells = grid.shape[0]
    nb_species = len(arr)
    # new_pmax = np.copy(pmax)
    
    if(monitoring):
        # setting collisions
        nb_groups = (nb_species*(nb_species+1))//2
        n = nb_cells*nb_groups
        collisions = np.zeros((n,3)) # ['cell_idx','° couples','quantity'] # per cell and colliding couples, thus : nb_cells x nb_species**2 lines
        # initializing collisions - we could do it outside ...
        groups_list = np.arange(nb_groups)
        for k in range(nb_cells):
            collisions[k*nb_groups:(k+1)*nb_groups] = k
            collisions[k*nb_groups:(k+1)*nb_groups,1] = groups_list

        # setting tracking
        tracking = np.zeros((nb_cells, 5)) # ['cell_idx','max_proba','mean_proba','mean_number_of_particles','mean_distance', thus nb_cells lines

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
            
            proba = probability(vr_norm = vr_norm, pmax = pmax[idx], cross_sections = cross_sections_couples) # TODO : at this point, the cross_section depends on the considered couple - this is the only thing that needs changing
            
            max_proba = np.max(proba)

            if(max_proba>1):
                pmax[idx] = max_proba*pmax[idx]
            
            collidings_couples = is_colliding(proba) # indexes in array of the ACTUALLY colliding couples

            if(monitoring):
                if(collidings_couples.shape[0]>0):
                    colliding_parts = array[collidings_couples]
                    # a 2D-array with [[[c1,i1],[c2,i2]], [[c3,i3],[c4,i4]], ...] ; c for container index, i for index in container 
                    # ideally, we would like to do a groupby on container indexes and then count the number 
                    # this is the use of group_fn, defined as a lambda function of 'arr' using first *set_groups* then *get_groups*
                    # and outside this function as it can be setup prior to simulation
                    groups = group_fn(colliding_parts.astype(int)) # once we have the groups we can count the quantity for each groups, and save it
                    unique, counts = np.unique(groups, return_counts=True) # so there we may not have all groups in unique
                    collisions[(idx*nb_groups+unique).astype(int),2] = counts

                    # tracking - we could add more than that easily
                    # TODO : it's a little counter intuitive to have average being updated outside this function but pmax inside.
                    # but it's ok
                    tracking[idx,:] = np.array([idx, pmax[idx], np.mean(proba[collidings_couples]),\
                        average[idx], np.mean(d[collidings_couples])])
                else:
                    tracking[idx,:] = np.array([idx, pmax[idx], np.nan,\
                                            average[idx], np.nan])

            array[collidings_couples] = reflect(array[collidings_couples], vr_norm[collidings_couples])

            for k in range(len(array)):
                c1, c2 = array[k,0], array[k,1]
                c = parts[k]
                arr[c[0,0]][c[0,1]][:] = c1 # copy
                arr[c[1,0]][c[1,1]][:] = c2

    if(monitoring):
        return collisions, tracking # in theory it is useless to return pmax


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

def index_choosen_couples(current, candidates, verbose = False): # per cell - I dont see how we can vectorize it as the number of candidates per cell depends on the cell.
    # in the future, it will be parallized so it should be ok.
    try :
        return np.random.randint(low = 0, high = current, size = (candidates,2)) # np.random.default_rng().choice(current, size = (candidates, 2), replace=False, axis = 1, shuffle = False) # we dont need shuffling
    except ValueError as e:
        if(verbose):
            print(e) 
            print(f'{current} < 2 x {candidates} => Some macro-particles will collide several times during the time step.')
        return np.random.randint(low = 0, high = current, size = (candidates,2)) # np.random.RandomState.choice(current, size = (candidates, 2), replace = True)
        # return np.random.default_rng().choice(current, size = (candidates, 2), replace=True, axis = 1, shuffle = False) # we dont need shuffling

def probability(vr_norm, pmax, cross_sections): # still per cell
    # vr_norm should be : np.linalg.norm((arr[choices][:,1,2:]-arr[choices][:,0,2:]), axis = 1)
    # returns a list of [True, False, etc.]
    return cross_sections/pmax*vr_norm
    # in theory, cross_sections is already present in pmax, so we could simplify it in the future (it is required though for different cross-sections and all)
    # returns an array of the size of len(choices) with the probability over each dimension

def is_colliding(proba):
    r = np.random.random(size = proba.shape)
    return np.where(proba>r)[0] # 1,0).astype(bool)
    # return the indexes where there is in fact collisions

def reflect(arr, vr_norm): # TODO : problem : here we suppose the mass is identical which is not the case 
    
    # reflection for an array containing the colliging couple
    # arr here is in fact arr[is_colliding(proba)] 
    # the colliding couples are already selected
    r = np.random.random(size = (2,arr.shape[0]))
    ctheta = 2*r[0,:]-1
    stheta = np.sqrt(1-ctheta*ctheta)
    phi = 2*np.pi*r[1,:]
    
    v_cm = 0.5*(arr[:,0,2:]+arr[:,1,2:]) # conserved quantity for same mass particles
    v_r_ = np.expand_dims(vr_norm, axis = 1)*np.stack((stheta*np.cos(phi), stheta*np.sin(phi), ctheta),  axis = 1) # 

    arr[:,0,2:] = v_cm + 0.5*v_r_
    arr[:,1,2:] = v_cm - 0.5*v_r_

    return arr


# default groups functions
def set_groups(n):
    groups = np.zeros((n, n))
    count = 0
    for i in range(n):
        for j in range(i, n):
            groups[i,j] = count
            groups[j,i] = count
            count +=1
    return groups

# now we want a function that returns the groups from the array
def get_groups(arr, groups):
    return groups[arr[:,0,0],arr[:,1,0]]