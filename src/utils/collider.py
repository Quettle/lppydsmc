import numpy as np
# TODO :
# * Test it for one type of particle
# * See formulas for various types of particles (mass, cross section) - it would need an adaptations as we have to store (pmax)pq for each p, q species. 
# It makes it complicated.
# Also note that there is a huge overhead calling python functions etc.
# so I should not DO everything like that but maybe include it in a bigger functions
# which would be much better

def handler_particles_collisions(arr, grid, currents, dt, average, pmax, cross_section, volume_cell, mr, remains, monitoring = True):
    # works in place for arr but may take very long ...
    # TODO : may return acceptance rates, and stuff like that...
    collisions = np.zeros(grid.shape)
    remains, cands = candidates(currents, dt, average, pmax, volume_cell, mr, remains)

    # new_pmax = np.copy(pmax)
    
    if(monitoring):
        monitor = np.array([0, 0]) # norm, proba

    for k, (i, j) in enumerate(np.ndindex(currents.shape)): # TODO : parallelize # looping over cells right now
        if(cands[i,j]>0):
            choice = index_choosen_couples(currents[i,j], int(cands[i,j]))
    
            g = grid[i,j]
            parts = np.array([[g[c[0]], g[c[1]]] for c in choice], dtype = int)
            array = np.array([[ arr[c[0,0]][c[0,1]] , arr[c[1,0]][c[1,1]] ] for c in parts])

            vr_norm = np.linalg.norm((array[:,1,2:]-array[:,0,2:]), axis = 1)
            d = np.linalg.norm((array[:,1,:2]-array[:,0,:2]), axis = 1)
            proba = probability(vr_norm = vr_norm, pmax = pmax[i,j], cross_sections = cross_section)

            if(monitoring): # summed over all the cells for now
                monitor = monitor + np.array([np.sum(d), np.sum(proba)])
            
            # TODO : should update pmax here (or return something)...
            max_proba = np.max(proba)
            if(max_proba>1):
                pmax[i,j] = max_proba*pmax[i,j]
            
            collidings_couples = is_colliding(proba)
            collisions[i,j]+=collidings_couples.shape[0]

            array[collidings_couples] = reflect(array[collidings_couples], vr_norm[collidings_couples])

            for k in range(len(array)):
                c1, c2 = array[k,0], array[k,1]
                c = parts[k]
                arr[c[0,0]][c[0,1]][:] = c1 # copy
                arr[c[1,0]][c[1,1]][:] = c2

    if(monitoring):
        return remains, collisions, pmax, monitor # in theory it is useless to return pmax
    else:
        return remains, collisions, pmax

def candidates(currents, dt, average, pmax, volume_cell, mr, remains):
    """ Returns the number of candidates couples to perform dsmc collisions between particles. Note that this formula is for one type of particle only.

    Args:
        currents (ndarray - 2D - float): number of particles per cell
        dt (float): time step
        average (ndarray - 2D - float): average number of particle in the cell
        pmax (ndarray - 2D - float): max probability per cell ()
        volume_cell (float or ndarray - 2D - float): volume of a cell
        mr (float): "macro-ratio" - ratio of real particles over macro-particles 

    Returns:
        (ndarray - 2D - float, ndarray - 2D - int) : the number of candidates to select per cell to perform collisions - fractional part first and then int part.
    """
    # for one type of particle for now
    remains, cands = np.modf(0.5*currents*average*pmax*mr/volume_cell*dt+remains) # (Nc Nc_avg mr (sigma vr)max dt)*/(2V_c)
    return remains, cands.astype(int) 

def index_choosen_couples(current, candidates, verbose = False): # per cell - I dont see how we can vectorize it as the number of candidates per cell depends on the cell.
    # in the future, it will be parallized so it should be ok.
    try :
        return np.random.default_rng().choice(current, size = (candidates, 2), replace=False, axis = 1, shuffle = False) # we dont need shuffling
    except ValueError as e:
        if(verbose):
            print(e) 
            print(f'{current} < 2 x {candidates} => Some macro-particles will collide several times during the time step.')
        return np.random.default_rng().choice(current, size = (candidates, 2), replace=True, axis = 1, shuffle = False) # we dont need shuffling

def probability(vr_norm, pmax, cross_sections): # still per cell
    # vr_norm should be : np.linalg.norm((arr[choices][:,1,2:]-arr[choices][:,0,2:]), axis = 1)
    # returns a list of [True, False, etc.]
    return cross_sections/pmax*vr_norm
    # in theory, cross_sections is already present in pmax, so we could simplify it in the future (it is required though for different cross-sections and all)
    # returns an array of the size of len(choices) with the probability over each dimension

def is_colliding(proba):
    r = np.random.random(size = proba.shape)
    return np.where(proba>r, 1,0).astype(bool)

def reflect(arr, vr_norm):
    
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
