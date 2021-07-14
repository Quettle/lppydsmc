from typing import ItemsView
import lppydsmc as ld
import numpy as np

def run(path_to_cfg):
    p = ld.config.cfg_reader.read(path_to_cfg) # parameters

    setup(p) # modify p in place

    # TODO :
    # - add SAVE
    # - add plot functions and params
    # - add tracking
    # - etc.

    # maybe some verbose and saving of the params 
    # simulation now
    simulate(p)

    return p


# ----------------- Simulation functions ---------------------- #


def simulate(p):
    df = pd.DataFrame(columns = ['x','y','vx','vy','vz','species']) # bucket for the particles - index of particles is the iteration number
    df_out_particles = pd.DataFrame(columns = ['x','y','vx','vy','vz','species'])
    nb_colls = np.zeros(grid.current.shape)
    collisions_with_walls = 0
    # df_collision_with_walls = pd.DataFrame(columns = ['x','y','type'])

    # adding particle before the simulation - step 0
    nb_species = params_species['count']
    containers = params_species['containers']
    types = params_species['types']
    types_dict = {}
    for idx, t in enumerate(types):
        types_dict[t] = idx
    arrays = [containers[k].get_array() for k in range(nb_species)]
    masses = [containers[k].mass() for k in range(nb_species)]

    # No injection
    # for arr, typ in zip(arrays, types):
    #   df = df.append(pd.DataFrame(data=np.concatenate((arr, [typ]*arr.shape[0]), axis = 1), index=[0]*arr.shape[0], columns = ['x','y','vx','vy','vz','species']))

    # defining useful arrays and ints 
        # injection
    remains = np.zeros((nb_species)) # fractionnal part of the number of particles to inject (it is then passed to the following time step)
        # grids
    averages = np.full(shape = grid.current.shape, fill_value = np.sum(mean_numbers_per_cell)) # average number of particles per cell
    cross_sections = params_species['cross_sections']
    pmax = 2*np.max(params_species['mean_speeds'])*np.mean(cross_sections)*np.ones(averages.shape) # max proba per cell in the simu
    remains_per_cell = np.zeros(shape = grid.current.shape, dtype = float) # remains per cell for the particles collisions step

    # SIMULATING
    print('|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|'.format(' it ', ' INIT ', ' INJECT ', ' DEL ', ' TRY', ' C. WALLS', ' C. PARTS' ))
    print('{:-^78}'.format(''))

    t = 0.0
    for it in range(1,iterations+1): # tqdm
        n1 = np.sum([containers[k].get_current() for k in range(nb_species)])
        
        # ------------------------- INJECTING PARTICLES -------------------------
        
        for k in range(nb_species):
            new, remains[k] = ld.injection.maxwellian(in_wall, in_vect, debits[k], vel_stds[k], dt, remains[k], drifts[k])
            containers[k].add_multiple(new)
            
        n2 = np.sum([containers[k].get_current() for k in range(nb_species)])
        
        # ---------------------------- PHASE : ADVECTING --------------------
            # MOVING PARTICLES
        arrays = [containers[k].get_array() for k in range(nb_species)]
        
        
        for k in range(nb_species):
            ld.advection.advect(arrays[k], update_functions[k], dt, t, args_update_functions[k], schemes[k]) # advect is inplace
        
            # HANDLING BOUNDARIES
        
        list_counts = []
        for k in range(nb_species):
            # initializing local variable
            arr = arrays[k]
            container = containers[k]
            
            count = np.full(fill_value = True, shape = arr.shape[0])
            idxes_out = []
            collided = []
            c = 0
            while(np.count_nonzero(count) > 0): # np.sum(count, where = count == True) > 0):
                c+=1
                ct, cp, cos_alpha = ld.advection.wall_collision.handler_wall_collision_point(arr[count], segments, a) # handler_wall_collision(arr[count], segments, a, radius)
                count, idxes_out_, cos_alpha = ld.advection.wall_collision.make_collisions_out_walls(arr, a, ct, cp, idx_out_walls, count, cos_alpha) # idxes_out : indexes of the particles (in arr) that got out of the system
                idxes_out.append(idxes_out_)

                # the first one that is received is the number of particles colliding with walls.
                if(c == 1):
                    collisions_with_walls += np.count_nonzero(count) # np.sum(count, where = count == True)
                    collided = np.copy(count) # np.where(collided[:collided_current], 1, 0)
                    
            if(len(idxes_out)>0):
                idxes_out = np.sort(np.concatenate(idxes_out))
                
                collided_current = collided.shape[0]
                
                for idx in np.flip(idxes_out): # view = constant time 
                    collided[idx] = collided[collided_current-1]
                    collided_current -= 1
                    
                list_counts.append(np.expand_dims(np.where(collided[:collided_current])[0], axis = 1))
                
                out_arr = container.pop_multiple(idxes_out)
                # using k instead of types[k] because this avoids using string in the dataframes which ALWAYS goes bad
                df_out_particles = df_out_particles.append(pd.DataFrame(data=np.concatenate((out_arr, np.expand_dims([k]*out_arr.shape[0], axis = 1)), axis = 1), index=[it]*out_arr.shape[0], columns = ['x','y','vx','vy','vz','species']))
            else:
                list_counts.append(np.array([])) # appending empty list to conserve the correspondance between rank in the global list and species
        

            
        particles_to_add = {}
        for k in range(nb_species):
            if types[k] in reactions:
                reacting_particles, particles_to_add_ = ld.advection.reactions.react(np.array(list_counts[k]), arrays = arrays, masses = masses, types_dict = types_dict, reactions = reactions[types[k]], p = None)
                # reacting_particles should be deleted in arrays[k]
                # particles_to_add should be added the the asssociated array
                list_counts[k] = reacting_particles # updating the actually reacting particles

                for key, val in particles_to_add_.items():
                    if(key in particles_to_add):
                        particles_to_add[key] += val
                    else:
                        particles_to_add[key] = val
            else:
                list_counts[k] = np.array([]) 
        # then and only then we delete everything in the update list_counts
        # print(f'Total collision with walls: {collisions_with_walls}')
        # print('DELETE')
        for k in range(nb_species): # here it's only one particle as it is colliding with the wall
            # thus it is easier to delete
            # print('{} - {}'.format(types[k], list_counts[k].shape[0]))
            containers[k].delete_multiple(list_counts[k])
            if(types[k] in particles_to_add):
                # print('ADDING - {} - {}'.format(types[k], len(particles_to_add[types[k]])))
                containers[k].add_multiple(np.array(particles_to_add[types[k]]))
        
        arrays = [containers[k].get_array() for k in range(nb_species)]
        
        grid.reset()
        for k in range(nb_species):
            arr = arrays[k]
            new_positions = ld.data_structures.grid.default_hashing(ld.data_structures.grid.pos_in_grid(arr[:,:2], resolutions, offsets, system_shape), res_y = resolutions[1])  
            parts_in_grid_format = ld.data_structures.grid.convert_to_grid_format(new = new_positions.shape[0], old = 0, container_idx = k)
            grid.add_multiple(new_positions, parts_in_grid_format)
    
        # ----------------------------- PHASE : DSMC COLLISIONS ----------------------------- 
            # TODO: make parallel (1st : note criticals functions in C++)    
        currents = grid.get_currents()
        averages = (it*averages+currents)/(it+1) # TODO: may be it too violent ? 

        remains_per_cell, nb_colls_, pmax, monitor = ld.collision.handler_particles_collisions(arrays, grid.get_grid(), currents, dt, averages, pmax, cross_sections, volume_cell, params_species['particles_weight'], remains_per_cell, monitoring = True)
        nb_colls += nb_colls_
        t += dt

        # ----------------------------- PLOTTING AND SAVING (OPTIONAL) ----------------------------- 
        if(it%adding_period == 0 or it == iterations):
            # print('Mean speed : ' + ' - '.join([str(np.mean(np.linalg.norm(arrays[k][:,2:], axis = 1))) for k in range(nb_species)]) + '\t m/s')
            for k, arr in enumerate(arrays):
                df = df.append(pd.DataFrame(data=np.concatenate((arr, np.expand_dims([k]*arr.shape[0], axis = 1)), axis = 1), index=[it]*arr.shape[0], columns = ['x','y','vx','vy','vz','species']))
                
        if(it%saving_period == 0 or it == iterations): # saving if last iteration too

            saver.save(it = it, append = {
                            'df' : df,
                            'collisions_per_cell' : nb_colls, # evolution of the number of collisions per cell - size : grid.shape[0] x grid.shape[1] (2D)
                            'total_distance' : float(monitor[0]), # evolution of the sum of the distance accross all cells 
                            'total_proba' : float(monitor[1]), # evolution of the sum of proba accross all cells
                            'pmax_per_cell' : pmax,  # evolution of the sum of pmax - per cell (2D)
                            'total_deleted' : len(idxes_out), # evolution of the number of deleted particles per cell (int)
                            'averages_per_cell' : averages, # evolution of the average number of particle per cell
                            'collisions_with_walls' : collisions_with_walls, # number of collisions with walls - evolution
                            'df_out_particles' : df_out_particles
                    })

            print('|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|'.format(it, n1, n2-n1, idxes_out.shape[0], c, collisions_with_walls, int(np.sum(nb_colls))))
            
            # resetting dataframe to not use too much memory
            collisions_with_walls = 0
            nb_colls = np.zeros(grid.current.shape)
            df = pd.DataFrame(columns = ['x','y','vx','vy','vz','species'])
            df_out_particles = pd.DataFrame(columns = ['x','y','vx','vy','vz','species'])
            
    saver.close()




# ------------------- Processing params ------------------- #

def setup(p):
    p['setup'] = {} # initialize a container that is going to contain the objects required for the setup phase.

    # converting points to segments (which are then sent to the system creator)
    points = [v for k, v in p['system']['points'].items()]
    p['system']['segments'] = ld.systems.helper.points_to_segments(points)

    p['setup']['system'] = ld.systems.creator.SystemCreator(p['system']['segments'])
    
    p['system']['offsets'] = p['setup']['system'].get_offsets()
    p['system']['system_shape'] = p['setup']['system'].system_shape()
    p['system']['a'] = p['setup']['system'].get_dir_vects()

    # species
    species = p['species']
    species['int_to_key'] = {str(i) : k for i, k in enumerate(p['species']['list'])} # attribute an integer to each species
    species['key_to_int'] = {v : int(k) for k, v in species['int_to_key'].items()}
    species['names'] = [k for k in species['key_to_int']]

    
    # params['cross_sections'] = cross_sections(radii, params['mean_speeds']) # np.array([params['containers'][k].get_params()[3] for k in range(params['count'])])
    # params['mean_free_paths'] = ld.utils.physics.mean_free_path(params['cross_sections'], np.sum(densities)) # taking the sum of all densities
    # params['mean_free_times'] = ld.utils.physics.mean_free_time(params['mean_free_paths'], v_mean = params['mean_speeds'])
    

    if(p['use_dsmc']):
        dsmc = p['dsmc']
        dsmc['grid']['resolutions'] = np.array(dsmc['grid']['resolutions'])
        dsmc['mnpc'] = np.array([v['mean_number_per_cell'] for k, v in dsmc['mean_number_per_cell'].items()])
        max_size = dsmc['grid']['max_size']

        # creating grid
        dsmc['cells_number'] = np.prod(dsmc['grid']['resolutions'])
        dsmc['cell_volume'] = p['system']['dz'] * np.prod(p['system']['system_shape'])/dsmc['cells_number']

        # setup useful objects
        p['setup']['grid'] = ld.data_structures.Grid(dsmc['cells_number'], max_size)
        p['setup']['dsmc_params'], p['setup']['containers'] = species_setup(p['species']['list'], init_number_per_cells = dsmc['mnpc'], number_of_cells = dsmc['cells_number'], cell_volume = dsmc['cell_volume'])

    if(p['use_fluxes']):
        fluxes = p['fluxes']
        pi1, pi2 = fluxes['pi1'], fluxes['pi2']
        fluxes['in_wall'] = np.array(pi1+pi2, dtype = float)
        fluxes['in_vect'] = np.array([pi2[1]-pi1[1],pi1[0]-pi2[0]], dtype = float) 
        fluxes['in_vect'] = fluxes['in_vect']/np.linalg.norm(fluxes['in_vect'])

        fluxes['names'] = [k for k in fluxes['species']]
        if(p['use_dsmc']):
            densities_dsmc = p['setup']['dsmc_params']['densities_dsmc']
            densities = np.array([densities_dsmc[species['key_to_int'][k]] for k in fluxes['names']])
        else:
            densities = np.array([species[k]['density'] for k in fluxes['names']])

        temperatures = np.array([v['temperature'] for k, v in fluxes['species'].items()])
        drifts = np.array([v['drift'] for k, v in fluxes['species'].items()])
        masses = np.array([species['list'][k]['mass'] for k in fluxes['names']])
        mean_speeds = ld.utils.physics.maxwellian_mean_speed(temperatures, masses)

        fluxes['debits'] = ld.utils.physics.maxwellian_flux(densities, mean_speeds)*np.linalg.norm(fluxes['in_wall'][:2]-fluxes['in_wall'][2:])*p['system']['dz']
        fluxes['vel_stds'] = ld.utils.physics.gaussian(temperatures, masses)
        fluxes['drifts'] = drifts
    
    if(p['use_reactions']):
        p['setup']['reactions'] = {}
        print(p['reactions']['walls'])
        reactions = ld.advection.reactions.parse(p['reactions']['walls'])
        
        p['setup']['reactions']['walls'] = ld.advection.reactions.parse(p['reactions']['walls'])

    if(p['use_poisson']):
        import lppydsmc.poisson_solver as ps
        poisson = p['system']['poisson_solver']
        p['setup']['mesh'] = ps.mesh.polygonal(poisson['mesh_resolution'],  np.flip(np.array(points), axis = 0), out_vertices_list = None) # polygonal recieves points in counter-clock order
        p['setup']['potential_field'], p['setup']['electric_field'] = ps.solver(p['setup']['mesh'], poisson['boundary_conditions'], poisson['charge_density'])

# ------------------ utils -------------------------- #

# useless
def init_max_proba(radii, mean_speeds, grid_shape):
    # radii and v_mean are of size Ns(number of species)
    # grid_shape is of size Nc (the number of cells)
    shape_pmax = [radii.shape[0], radii.shape[0]] + list(grid_shape)
    pmax = np.ones(tuple(shape_pmax), dtype = float)
    cross_sections = np.ones(tuple(shape_pmax), dtype = float)

    for i in range(pmax.shape[0]):
        for j in range(i+1):
            if(i==j):
                cross_sections[i,j] = np.pi * 4*radii[i]**2
                pmax[i,j] *= 2*mean_speeds[i] * np.pi * 4*radii[i]**2  # we'll take the max proba straight away ?
            else:
                cross_sections[i,j] = np.pi * (radii[i]+radii[1])**2 
                pmax[i,j] *= np.abs(mean_speeds[i]-mean_speeds[j]) * np.pi * (radii[i]+radii[1])**2   # we'll take the max proba straight away ?
    return pmax, cross_sections

def cross_sections(radii, mean_speeds):
    # radii and v_mean are of size Ns(number of species)
    # grid_shape is of size Nc (the number of cells)
    shape_out = [radii.shape[0], radii.shape[0]]
    cross_sections = np.ones(tuple(shape_out), dtype = float)

    for i in range(cross_sections.shape[0]):
        for j in range(i+1):
            if(i==j):
                cross_sections[i,j] = np.pi * 4*radii[i]**2
            else:
                cross_sections[i,j] = np.pi * (radii[i]+radii[1])**2 
                cross_sections[j,i] = cross_sections[i,j]
    return cross_sections

def convert(species):
    """
    Convert from dictionnary of species to dictionnary of quantities.
    """
    dico = {}
    
    types = []
    densities = []
    charges = []
    masses = []
    radii = []
    
    for key, val in species.items():
        types.append(key)
        densities.append(val['density'])
        charges.append(val['charge'])
        masses.append(val['mass'])
        radii.append(val['radius'])
        
        
    dico['types'] = types
    dico['densities'] = np.array(densities)
    dico['charges'] = np.array(charges)
    dico['masses'] = np.array(masses)
    dico['radii'] = np.array(radii)
    
    # count of species
    dico['count'] = len(types)
    
    return dico

def species_setup(species, init_number_per_cells, number_of_cells, cell_volume):
    params = convert(species)
    
    types = params['types']
    densities = params['densities']
    charges = params['charges']
    masses = params['masses']
    radii = params['radii']
    
    # computed quantities
    size_arrays = init_number_per_cells*number_of_cells # max size for the array
    containers = [ld.data_structures.Particle(types[k], charges[k], masses[k], radii[k], size_arrays[k]) for k in range(params['count'])]
    
    params['total_number_particles_simu'] = init_number_per_cells * number_of_cells
    params['total_number_particles_real'] = cell_volume * densities * number_of_cells 
    params['particles_weight'] = params['total_number_particles_real']/params['total_number_particles_simu']
    params['particles_weight'] = params['particles_weight'].astype(int)
    w0 = params['particles_weight'][0]
    for w in params['particles_weight']:
        assert(w0==w)
    params['particles_weight'] = w0
    params['densities_dsmc'] = densities/params['particles_weight']

    return params, containers