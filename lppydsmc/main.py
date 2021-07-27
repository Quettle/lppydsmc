import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os

import lppydsmc as ld

def main(path_to_cfg, save=True):
    p = ld.config.cfg_reader.read(path_to_cfg) # parameters
    setup(p) # modify p in place

    # add option to remove the previous directory if it exists (because it can create issues with the simulations)
    # Note : it should depends on the option of the user
    # if loading particles, for example, may be we don't want to remove what was previously here.

    if(not os.path.isdir(p['setup']['path'])):
        os.makedirs(p['setup']['path']) 

    if(save): # savnig the new params to a file so the user can go and debug it or refer to all the simulations params when needed.
        from configobj import ConfigObj
        pp_dict = convert_objects(p.dict())
        pp = ConfigObj(pp_dict)
        pp.filename = '{}/{}.ini'.format(p['setup']['path'], 'params')
        pp.write()
        
    # dealing with seeding
    np.random.seed(p['simulation']['seed'])

    # TODO :
    # - add plot functions and params
    # - add more complexe system (with parts of the system that we dont take) - example of the cylinder.
    # - maybe some verbose and saving of the params 
    
    # SIMULATION
    if(p['use_monitoring']): # TODO : may be this should be done in the setup phase
        saver = ld.data.saver.Saver(p['setup']['path'], 'monitoring.h5') # TODO : maybe do something with the path

        # plotting even though it's temporary 
        if(not os.path.exists(p['setup']['path']/'images/')):
            os.makedirs(p['setup']['path']/'images/')
            
        with saver.__enter__() as store : # ensuring saver is closed in the end, no matter the termination
            monitor_dict = None
            monitor = init_monitor(p['use_fluxes'], p['use_dsmc'], p['use_reactions'])

            monitor_dict = {
                'monitor' : monitor,
                'period_adding' : p['monitoring']['period_adding'],
                'period_saving' : p['monitoring']['period_saving'],
                'saver' : saver,
            }
            simulate(p['use_fluxes'], p['use_dsmc'], p['use_reactions'], p['use_plotting'], p['use_verbose'], monitor_dict = monitor_dict, **p['setup'])

    else:
        simulate(p['use_fluxes'], p['use_dsmc'], p['use_reactions'], p['use_plotting'], p['use_verbose'], monitor_dict = None, **p['setup'])

    print('SIMULATION FINISHED')
    print('Path to results : {}'.format(p['setup']['path']))

# ----------------- convert ------------------- #
def convert_objects(p):
    pp = {}
    for k, v in p.items():
        if(type(v) is dict):
            pp[k] = convert_objects(v)
        elif(type(v) is list):
            for i in range(len(v)):
                v[i] = convert_object(v[i])
            pp[k] = v
        else:
            pp[k] = convert_object(v)
    return pp

def convert_object(o):
    try :
        return '{}{}'.format(o.__name__, inspect.signature(o))
    except Exception:
        try :
            return o.__str__()
        except Exception:
            return o
# ----------------- Simulation functions ---------------------- #

def simulate(use_fluxes, use_dsmc, use_reactions, use_plotting, use_verbose, monitor_dict, **kwargs):

    # ------- Final setup ---------- #
    species = kwargs['species'] # dict
    nb_species = len(species)
    system = kwargs['system']
    points = kwargs['points']
    containers = kwargs['containers'] # dict
    cross_sections_matrix = kwargs['cross_sections_matrix']
    path = kwargs['path']
    reflect_fns = kwargs['reflect_fns']
    user_generic_args = kwargs['user_generic_args'] # list
    user_defined_args = kwargs['user_defined_args'] # dict

    # initialization of particles
    particles_initialization = kwargs['particles_initialization']
    particles_in_system = 0
    for specie, params in particles_initialization.items():
        containers[specie].add_multiple(ld.initialization.particles.initialize(np.array(points), params['quantity'], params['type'], params['params']))
        particles_in_system+=params['quantity']

    # simulation    
    time_step = kwargs['time_step'] # float
    iterations = kwargs['iterations'] # int
    masses = np.array([container.mass() for specie, container in containers.items()])

    # advect
    update_functions = kwargs['update_functions']
    args_update_functions = kwargs['args_update_functions']
    schemes = kwargs['schemes']

    if(use_fluxes):
        injected_species = kwargs['inject']['species'] # list is enough here I think
        idx_injected_species = [species[s] for s in injected_species]
        in_wall = kwargs['inject']['in_wall']
        in_vect = kwargs['inject']['in_vect']
        debits = kwargs['inject']['debits']
        vel_stds = kwargs['inject']['vel_stds']
        drifts = kwargs['inject']['drifts']
        
        remains = np.zeros((len(injected_species))) # finally it's nb_species as we could also add the output species

    if(use_dsmc):
        grid = kwargs['dsmc']['grid']
        resolutions = kwargs['dsmc']['resolutions']
        max_proba = kwargs['dsmc']['max_proba']
        cell_volume = kwargs['dsmc']['cell_volume']
        particles_weight = kwargs['dsmc']['particles_weight']
        use_same_mass = kwargs['dsmc']['use_same_mass']
        if(use_same_mass):
            dsmc_masses = None
        else:
            dsmc_masses = masses

        # useful values
        remains_per_cell = np.zeros(shape = grid.current.shape, dtype = float)
        averages = np.full(shape = grid.current.shape, fill_value = np.sum(kwargs['dsmc']['mean_numbers_per_cell']))

    if(use_reactions):
        boundaries_reactions = kwargs['reactions']['boundaries']
   
    disable = False

    if(use_verbose):
        from time import time
        disable = True
        period_verbose = kwargs['verbose']['period']
        print('|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|'.format(' it ', ' INIT ', ' INJECT ', ' DEL ', ' C. WALLS ', ' C. PARTS ', ' REACT ', ' EXEC TIME (s) '))
        print('{:-^94}'.format(''))
        

    if(use_plotting):
        import matplotlib.pyplot as plt 
        fig, ax = plt.subplots(constrained_layout = True)
        if(kwargs['plotting']['plot_distribution']):
            plot_distribution = kwargs['plotting']['plot_distribution']
            fig2, axes = plt.subplots(2,1,constrained_layout = True)
        period_plotting = kwargs['plotting']['period']

    # Monitoring    
    monitoring = False
    group_fn = None

    if(monitor_dict is not None):
        monitoring = True # variable that will change value as the simulation goes - it will follow a period of 'period_adding'
        monitor = monitor_dict['monitor']
        period_saving = monitor_dict['period_saving']
        period_adding = monitor_dict['period_adding']
        saver = monitor_dict['saver']
        if(use_fluxes):
            fluxes_arr = np.zeros((nb_species,3))
            for k in range(nb_species):
                fluxes_arr[k,2] = k # setting the species index

        if(use_dsmc):
            groups = ld.collision.collider.set_groups(nb_species)
            group_fn = lambda arr : ld.collision.collider.get_groups(arr, groups)

    current_time = 0.   
    # SIMULATING HERE  

    nb_out_particles = 0
    nb_init_particles = particles_in_system
    nb_injected_particles = 0 if use_fluxes else 'NA'
    nb_collisions_with_walls = 0
    nb_collisions_dsmc = 0 if use_dsmc else 'NA'
    nb_reactions = 0 if use_reactions else 'NA'
    exec_time = 0

    for iteration in tqdm(range(1, iterations+1), disable=disable):

        if(use_verbose and (iteration-1)%period_verbose==0): # plot for the previous iterations
            if(iteration-1!= 0): verbose_time = '{:.3e}'.format((time()-exec_time)/period_verbose)
            else : verbose_time = 0
            print('|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^15}|'.format(iteration-1, nb_init_particles, nb_injected_particles, nb_out_particles, \
                nb_collisions_with_walls, nb_collisions_dsmc, nb_reactions, verbose_time))
            exec_time = time() # TODO : ignore monitoring exec time ?

        # right time to monitor
        if(monitor_dict is not None and iteration%period_adding == 0): monitoring = True

        if(use_fluxes):
            inject_qties = inject(injected_species, containers, in_wall, in_vect, debits, vel_stds, time_step, remains, drifts)
            
            nb_injected_particles = np.sum(inject_qties)
            nb_init_particles += nb_injected_particles
            if(monitoring):
                fluxes_arr[idx_injected_species, 0] = inject_qties # saved right after (waiting for out-particles)
                

        advect([container.get_array() for specie, container in containers.items()], current_time, time_step, update_functions, args_update_functions, schemes)

        results_reflect_particles = reflect_out_particles(containers, system, reflect_fns, user_generic_args, user_defined_args, monitoring) 
        # if monitoring is True then return idx_gsi and out_particles (the particles that got out)
        nb_out_particles = results_reflect_particles[1]
        nb_init_particles -= nb_out_particles
        nb_collisions_with_walls = sum([len(collidings) for collidings in results_reflect_particles[0]])
        if(monitoring):
            if(use_fluxes):
                # out particles
                for idx, out_parts in enumerate(results_reflect_particles[2]):
                    if(out_parts != []):
                        array = np.concatenate((out_parts, np.full((out_parts.shape[0],1), idx)), axis = 1) 
                        monitor['out_particles'] = monitor['out_particles'].append(pd.DataFrame(array, index = np.full(array.shape[0], iteration), \
                            columns = monitor['out_particles'].columns))
                        fluxes_arr[idx, 1] = out_parts.shape[0]
                    else:
                        fluxes_arr[idx, 1] = 0
                monitor['fluxes'] = monitor['fluxes'].append(pd.DataFrame(fluxes_arr, index=[iteration]*fluxes_arr.shape[0], \
                        columns = monitor['fluxes'].columns))

            # colliding particles
            colliding_particles_positions = []
            for k, s in enumerate(species):
                # at this point, we can have particles that got deleted, they should not be accounted for
                # /!\ results_reflect_particles[0][k] has to be a list, else the numpy does not understand it !
                coll_parts_ = containers[s].get(results_reflect_particles[0][k])[:,:2] # particles that collided with a wall, per species
                # in the end we are going to add them one after the others in a 1D-array (well infact total_nb_collisions x 3 - 3 for (x, y, species))
                if(coll_parts_.shape[0]>0):
                    colliding_particles_positions.append(np.concatenate((coll_parts_, np.full(shape = (coll_parts_.shape[0],1), fill_value = k)),\
                        axis = 1)) # only x and y
            if(colliding_particles_positions != []):
                colliding_particles_positions = np.concatenate(colliding_particles_positions) # not sure this works
                if(not use_reactions): # in the case we use reactions, we then add the reaction for each particle (if ones happens that is)
                    monitor['wall_collisions'] = monitor['wall_collisions'].append(pd.DataFrame(colliding_particles_positions, \
                         index=[iteration]*colliding_particles_positions.shape[0], columns = monitor['wall_collisions'].columns))

        if(use_reactions):
            nb_reactions, happening_reactions_relative_indexes = recombine(results_reflect_particles[0], \
                containers, boundaries_reactions, masses, species, monitoring) # None if monitoring is False
                # if monitoring is not False, then happening_reactions_relative_indexes is for now a list of 1D-arrays of different sizes

            if(monitoring):
                # happening_reactions_relative_indexes is an array that contains the idx of the reactions and -1 if it did not react
                if(colliding_particles_positions != []):
                    happening_reactions_relative_indexes = np.concatenate(happening_reactions_relative_indexes, axis = 0)
                    # colliding_particles_positions shape : total_nb_collisions x 3
                    # happening_reactions_relative_indexes : total_nb_collisions (we need to expand dims in the axis = 1)
                    # then concatenate along that axis
                    monitor['wall_collisions'] = monitor['wall_collisions'].append(pd.DataFrame(np.concatenate((colliding_particles_positions, \
                        np.expand_dims(happening_reactions_relative_indexes, axis = 1)), axis = 1),\
                        index=[iteration]*colliding_particles_positions.shape[0], \
                            columns = monitor['wall_collisions'].columns))

        if(use_dsmc):
            # try :
            results_dsmc = dsmc(containers, grid, resolutions, system.get_offsets(), system.get_shape(), \
                averages, iteration, time_step, max_proba, cell_volume, particles_weight, cross_sections_matrix,\
                    remains_per_cell, dsmc_masses, monitoring, group_fn)

            # except Exception as e:
            #     ax.clear()
            #     plot_system(ax, containers, system)
            #     plt.savefig(path/'images'/f'{iteration}.png', dpi = 300)
            #     plt.show()
            #     raise e
            if(monitoring):
                nb_collisions_dsmc = results_dsmc[0]

                monitor['dsmc_collisions'] = monitor['dsmc_collisions'].append(pd.DataFrame(results_dsmc[1], \
                    index=[iteration]*results_dsmc[1].shape[0], columns = monitor['dsmc_collisions'].columns))
                monitor['dsmc_tracking'] = monitor['dsmc_tracking'].append(pd.DataFrame(results_dsmc[2], \
                    index=[iteration]*results_dsmc[2].shape[0], columns = monitor['dsmc_tracking'].columns))

        current_time+=time_step
    
        if(use_plotting and iteration%period_plotting==0):
            ax.clear()
            plot_system(ax, containers, system)
            fig.savefig(path/'images'/f'{iteration}.png', dpi = 300)

            if(plot_distribution):
                axes[0].clear()
                axes[1].clear()
                plot_velocity_distribution(axes, containers)
                fig2.savefig(path/'images'/f'distrib_{iteration}.png', dpi = 300)

            # maybe add more than that ...

        if(monitoring):
            # adding particles
            for k, (specie, container) in enumerate(containers.items()):
                arr = container.get_array()
                array = np.concatenate((arr, np.full((arr.shape[0],1), k)), axis = 1) 
                # we prefer saving with k, this way everything is set to int / float 
                # which makes it easier for pandas.
                # Alternatively, we could save specie
                # however it caused issues when analyzing in the past
                monitor['particles'] = monitor['particles'].append(pd.DataFrame(array, index = np.full(array.shape[0], iteration), \
                    columns = monitor['particles'].columns))


        if(monitoring and iteration%period_saving == 0):
            # print('{} : {}'.format(iteration, monitor['wall_collisions']))  
            # for key, value in monitor.items():
            #     print(key)
            #     pprint(value)
            saver.save(it = iteration, append = monitor)
            
            # reinitializing monitor
            monitor = init_monitor(use_fluxes, use_dsmc, use_reactions)

        monitoring = False
    
# ---------------- Plotting ---------------- #
def plot_system(ax, containers, system): # add it to the plotting tools may be wiser ...
    for segment in system.get_segments():
        ax.plot(segment[[0,2]], segment[[1,3]], color = 'k')

    for specie, container in containers.items():
        arr = container.get_array()
        ax.scatter(arr[:,0], arr[:,1], label = specie, s = 0.1)
    ax.legend(loc='best')
    ax.axis('equal')

def plot_velocity_distribution(axes, containers):
    vx, vy = [], []
    species = []
    for specie, container in containers.items():
        arr = container.get_array()
        vx.append(arr[:,2])
        vy.append(arr[:,3])
        species.append(specie)

    axes[0].hist(vx, label = species, bins = 'auto', stacked = True, histtype = 'barstacked')
    axes[1].hist(vy, label = species, bins = 'auto', stacked = True, histtype = 'barstacked')
    
    axes[0].legend(loc='best')
    axes[1].legend(loc='best')

    axes[0].set_xlabel('vx (m/s)')
    axes[1].set_xlabel('vy (m/s)')

def init_monitor(use_fluxes, use_dsmc, use_reactions): # 1-layer dictionnary to make saving easier
    monitor = {}
    monitor['particles'] = pd.DataFrame(columns = ['x','y','vx','vy','vz','species']) # save all particles, index is the iteration
    monitor['wall_collisions'] = pd.DataFrame(columns = ['x','y','species'])

    if(use_fluxes):
        monitor['fluxes'] = pd.DataFrame(columns = ['in','out','species'])
        monitor['out_particles'] = pd.DataFrame(columns = ['x','y','vx','vy','vz','species'])
    if(use_dsmc):
        monitor['dsmc_collisions'] = pd.DataFrame(columns = ['cell_idx','idx_reactions','quantity'])
        monitor['dsmc_tracking'] = pd.DataFrame(columns = ['cell_idx','max_proba','mean_proba','mean_number_of_particles','mean_distance']) # the 3 last fields are for particles that collided !
        # max_proba is the max_proba used to "normalize" the probabilities

    if(use_reactions):
        monitor['wall_collisions'] = pd.DataFrame(columns = ['x','y','species','reaction'])
    else:
        monitor['wall_collisions'] = pd.DataFrame(columns = ['x','y','species'])

    return monitor

# ------------------- Inject -------------------- #
def inject(species, containers, in_wall, in_vect, debits, vel_stds, dt, remains, drifts): # in place for remains because numpy array.
    # some species may not be injected while others will /!\
    remains[:], inject_qties = ld.injection.get_quantity(debits, remains, dt) # the [:] makes sure it's in place
    for i, k in enumerate(species): # idx, key (example : 0, 'I')
        new = ld.injection.maxwellian(in_wall, in_vect, vel_stds[i], inject_qties[i], dt, drifts[i])
        containers[k].add_multiple(new)
    return inject_qties

# ----------------- Advect ----------------------- #
def advect(arrays, time, time_step, update_functions, args_update_functions, schemes): # in place
    for k in range(len(arrays)):
        ld.advection.advect(arrays[k], update_functions[k], time_step, time, args_update_functions[k], schemes[k]) # advect is inplace 

# --------------- Handle boundaries -------------- #

def reflect_out_particles(containers, system, reflect_fns, user_generic_args, user_defined_args, monitoring = None): # will also delete particles that got out of the system.
    segments = system.get_segments()
    idx_out_segments = system.get_idx_out_segments()

    idxes_gsi = []

    # could be defined in a higher function but it's little overhead so it's ok.
    generic_args = {
        'cp':None, # depends on the particles
        'ct':None,
        'index_walls': None,
        'mass' : None,
        'directing_vectors': system.get_dir_vects(),
        'normal_vectors': system.get_normal_vectors(),
    }

    nb_out_particles = 0

    if(monitoring): out_particles = []
    for k, (specie, container) in enumerate(containers.items()):
        # initializing local variable
        arr = container.get_array()
        container = containers[specie]
        generic_args['mass'] = container.mass()
        absolute_colliding_bool = np.full(fill_value = True, shape = arr.shape[0])
        absolute_idxes_exited = []
        collided = []
        c = 0
        
        while(np.count_nonzero(absolute_colliding_bool) > 0): # np.sum(count, where = count == True) > 0):
            c+=1    
            ct, cp = ld.advection.boundaries.get_possible_collisions(arr[absolute_colliding_bool], segments, generic_args['directing_vectors']) # ct, cp - size of arr[count]
            # colliding_particles and idxes_walls is size of ct meaning arr[count], while idxes_out_ only contains the idxes of the particles that exited the system
            relative_colliding_bool, relative_idxes_exiting, idxes_walls = ld.advection.boundaries.get_relative_indexes(ct, idx_out_segments)
            # here we need all to have the same shape.
            generic_args['cp'] = cp
            generic_args['ct'] = ct
            generic_args['index_walls'] = idxes_walls
            absolute_idxes_exited_ = ld.advection.boundaries.get_absolute_indexes(absolute_colliding_bool, relative_colliding_bool, relative_idxes_exiting)
            # arr require the absolute indexes, while idxes_walls, ct and cp require relative_colliding_bool
            ld.advection.boundaries.reflect_back_in(arr, absolute_colliding_bool, relative_colliding_bool, generic_args, reflect_fns[k], user_generic_args[k], user_defined_args[k]) # in place

            absolute_idxes_exited.append(absolute_idxes_exited_)

            # the first one that is received is the number of particles colliding with walls.
            if(c == 1):
                collided = np.copy(absolute_colliding_bool) # np.where(collided[:collided_current], 1, 0)
                
        if(len(absolute_idxes_exited)>0):
            absolute_idxes_exited = np.sort(np.concatenate(absolute_idxes_exited))
            
            collided_current = collided.shape[0]
            
            for idx in np.flip(absolute_idxes_exited): # view = constant time 
                collided[idx] = collided[collided_current-1]
                collided_current -= 1
                
            # in order not to select particles that got out of the system
            # idxes_gsi.append(list(np.expand_dims(np.where(collided[:collided_current])[0], axis = 1)))
            idxes_gsi.append(list(np.where(collided[:collided_current])[0]))

            absolute_idxes_exited_ = container.pop_multiple(absolute_idxes_exited)
            nb_out_particles+=absolute_idxes_exited_.shape[0]
            if(monitoring): out_particles.append(absolute_idxes_exited_)

        else:
            idxes_gsi.append([]) # appending empty list to conserve the correspondance between rank in the global list and species
            if(monitoring): out_particles.append([])
    if(monitoring):
        return idxes_gsi, nb_out_particles, out_particles, 

    return idxes_gsi, nb_out_particles

def recombine(idxes_gsi, containers, reactions, masses, species, monitoring = False): # catalytic boundary recombination - inplace 
    particles_to_add = {}
    arrays = [container.get_array() for specie, container in containers.items()]
    
    count_reactions = 0

    if(monitoring): happening_reactions_relative_indexes = []

    for k, type_part in enumerate(containers):
        if type_part in reactions: # all particles colliding are not necessarily in reactions !
            idx_colliding_particles = np.expand_dims(np.array(idxes_gsi[k]), axis = 1)
            results = ld.advection.reactions.react(idx_colliding_particles, arrays = arrays,\
                 masses = masses, types_dict = species, reactions = reactions[type_part], p = None, monitoring = monitoring)
            # results = (np.array(reacting_particles), particles_to_add, happening_reactions) if monitoring, else (np.array(reacting_particles), particles_to_add)
            # reacting_particles should be deleted in arrays[k]
            # particles_to_add should be added the the asssociated array
            idxes_gsi[k] = results[0] # updating the actually reacting particles
            count_reactions.append(len(results[0]))
            if(monitoring):
                happening_reactions_relative_indexes.append(results[2])

            for key, val in results[1].items():
                if(key in particles_to_add):
                    particles_to_add[key] += val
                else:
                    particles_to_add[key] = val
        else:
            if(monitoring): happening_reactions_relative_indexes.append(np.zeros((len(idxes_gsi[k])))) # adding particles there even if they dont react
            idxes_gsi[k] = np.array([]) 

    # then and only then we delete everything in the update idxes_gsi
    for k, (type_part, container) in enumerate(containers.items()): # here it's only one particle as it is colliding with the wall
        container.delete_multiple(idxes_gsi[k])
        if(type_part in particles_to_add):
            containers[type_part].add_multiple(np.array(particles_to_add[type_part]))

    if(monitoring): return count_reactions, happening_reactions_relative_indexes # reactions actually happening in the idxes_gsi
                                                                # happening_reactions_relative_indexes contains a list of array (1 array = 1 species)
                                                                # containing, for each colliding particles, a number between 0 and nb_rections for the given species
                                                                # 0 being no-reaction

# --------------- dmsc -------------- #

def dsmc(containers, grid, resolutions, system_offsets, system_shape, averages, iteration, time_step, \
    max_proba, cell_volume, particles_weight, cross_sections, remains_per_cell, masses = None, monitoring = False, group_fn = None):
    arrays = [container.get_array() for specie, container in containers.items()]
    
    grid.reset()
    for k in range(len(arrays)):
        arr = arrays[k]
        new_positions = ld.data_structures.grid.default_hashing(ld.data_structures.grid.pos_in_grid(arr[:,:2], \
            resolutions, system_offsets, system_shape), res = resolutions[1])  
        parts_in_grid_format = ld.data_structures.grid.convert_to_grid_format(new = new_positions.shape[0], old = 0,\
             container_idx = k)
        grid.add_multiple(new_positions, parts_in_grid_format)
 
    # ----------------------------- PHASE : DSMC COLLISIONS ----------------------------- 
        # TODO: make parallel (1st : note criticals functions in C++)    
    currents = grid.get_currents()
    averages = (iteration*averages+currents)/(iteration+1) # TODO: may be it too violent ? 

    results = ld.collision.handler_particles_collisions(arrays, grid.get_grid(), currents, time_step, \
            averages, max_proba, cross_sections, cell_volume, particles_weight, remains_per_cell, masses, monitoring = monitoring, group_fn = group_fn) # inplace for remains_per_cell
    # count_collisions if monitoring = False, else results = count_collisions, collisions, tracking 
    return results
# ------------------- Processing params ------------------- #

def setup(p):
    p['directory'] = (Path(p['directory'])).resolve()

    # converting points to segments (which are then sent to the system creator)
    points = [v for k, v in p['system']['points'].items()]  

    p['system']['segments'] = ld.systems.helper.points_to_segments(points)
    system = ld.systems.creator.SystemCreator(p['system']['segments'], p['system']['out_boundaries']['out_boundaries'])
    p['system']['offsets'] = system.get_offsets()
    p['system']['system_shape'] = system.get_shape()
    p['system']['a'] = system.get_dir_vects()

    # species
    species = p['species']
    species['int_to_key'] = {str(i) : k for i, k in enumerate(p['species']['list'])} # attribute an integer to each species
    species['key_to_int'] = {v : int(k) for k, v in species['int_to_key'].items()}
    species['names'] = [k for k in species['key_to_int']]
    params_species = convert(p['species']['list'])

    volume = ld.utils.estimation.estimate_surface(samples = int(1e4), points = points)[0]*p['system']['dz']
    safety_factor = 2
    size_arrays = safety_factor * params_species['densities'] * volume # volume is approximated using a Monte Carlo method.
    
    cross_sections_matrix = get_cross_sections(params_species['radii']) # hard sphere model

    if(not p['use_fluxes'] and not p['use_particles_initialization']):
        print('You either need particles in the system or have an injection to launch a simulation.')
        raise ValueError

    if(p['use_dsmc']):
        dsmc = p['dsmc']
        dsmc['grid']['resolutions'] = np.array(dsmc['grid']['resolutions'])
        dsmc['mean_numbers_per_cell'] = np.array([v['mean_number_per_cell'] for k, v in dsmc['mean_number_per_cell'].items()])
        max_size = dsmc['grid']['max_size']

        # creating grid 
        dsmc['cells_number'] = np.prod(dsmc['grid']['resolutions'])
        dsmc['cell_volume'] = p['system']['dz'] * np.prod(p['system']['system_shape'])/dsmc['cells_number']

        dsmc['total_number_particles_simu'] = dsmc['mean_numbers_per_cell'] * dsmc['cells_number']
        dsmc['total_number_particles_real'] = dsmc['cell_volume'] * params_species['densities'] * dsmc['cells_number'] 
        dsmc['particles_weight'] = dsmc['total_number_particles_real']/dsmc['total_number_particles_simu']
        dsmc['particles_weight'] = dsmc['particles_weight'].astype(int)
        w0 = dsmc['particles_weight'][0]
        for w in dsmc['particles_weight']:
            assert(w0==w)
        dsmc['particles_weight'] = w0
        dsmc['densities_dsmc'] = params_species['densities']/dsmc['particles_weight']

        # adapting the size of the arrays for when creating all the containers
        size_arrays = size_arrays/dsmc['particles_weight'] 
    
    size_arrays = size_arrays.astype(int)
        
    if(p['use_fluxes']):
        fluxes = p['fluxes']
        pi1, pi2 = fluxes['pi1'], fluxes['pi2']
        fluxes['in_wall'] = np.array(pi1+pi2, dtype = float)
        fluxes['in_vect'] = np.array([pi2[1]-pi1[1],pi1[0]-pi2[0]], dtype = float) 
        fluxes['in_vect'] = fluxes['in_vect']/np.linalg.norm(fluxes['in_vect'])

        fluxes['names'] = [k for k in fluxes['species']]
        
        if(p['use_dsmc']):
            densities_dsmc = dsmc['densities_dsmc']
            densities = np.array([densities_dsmc[species['key_to_int'][k]] for k in fluxes['names']])
        else:
            densities = np.array([species['list'][k]['density'] for k in fluxes['names']])

        temperatures = np.array([v['temperature'] for k, v in fluxes['species'].items()])
        drifts = np.array([v['drift'] for k, v in fluxes['species'].items()])
        masses = np.array([species['list'][k]['mass'] for k in fluxes['names']])
        mean_speeds = ld.utils.physics.maxwellian_mean_speed(temperatures, masses)

        fluxes['debits'] = ld.utils.physics.maxwellian_flux(densities, mean_speeds)*np.linalg.norm(fluxes['in_wall'][:2]-fluxes['in_wall'][2:])*p['system']['dz']
        fluxes['vel_stds'] = ld.utils.physics.gaussian(temperatures, masses)
        fluxes['drifts'] = drifts

    if(p['use_reactions']):
        pass

    if(p['use_poisson']):
        import lppydsmc.poisson_solver as ps
        poisson = p['system']['poisson_solver']

    
    # /!\ Setting the parameters used in the simulation.
    p['setup'] = {}
    p['setup']['system'] = system
    p['setup']['points'] = points
    p['setup']['species'] = species['key_to_int']
    p['setup']['cross_sections_matrix'] = cross_sections_matrix
    p['setup']['path'] = p['directory']/p['name']
        # reflection on boundaries - reflect_fns
    p['setup']['reflect_fns'], p['setup']['user_generic_args'], p['setup']['user_defined_args'] = \
        reflection_functions_setup(p['system']['reflect_fns'], p['setup']['species'])
        # simulation
    p['setup']['time_step'] = p['simulation']['time_step'] # float
    p['setup']['iterations'] = p['simulation']['iterations'] # int
        # containers
    types = params_species['types']
    charges = params_species['charges']
    masses = params_species['masses']
    radii = params_species['radii']
    p['setup']['containers'] = {types[k] : ld.data_structures.Particle(types[k], charges[k], masses[k], radii[k], size_arrays[k]) for k in range(params_species['count'])}
        # particles initialization
    densities_ = densities if not p['use_dsmc'] else dsmc['densities_dsmc']
    p['setup']['particles_initialization'] = particles_initialization_setup(p['species']['initialization'], species['key_to_int'], masses, quantities = densities_*volume) 
        # verbose
    if(p['use_verbose']):
        p['setup']['verbose'] = {}
        p['setup']['verbose']['period'] = p['verbose']['period']
        # plotting
    if(p['use_plotting']):
        p['setup']['plotting'] = {}
        p['setup']['plotting']['period'] = p['plotting']['period']
        p['setup']['plotting']['plot_distribution'] = p['plotting']['plot_distribution']

        # dsmc  
    if(p['use_dsmc']):
        p['setup']['dsmc'] = {}
        p['setup']['dsmc']['grid'] = ld.data_structures.Grid(dsmc['cells_number'], max_size)
        p['setup']['dsmc']['resolutions'] = dsmc['grid']['resolutions']
        p['setup']['dsmc']['use_same_mass']  = dsmc['use_same_mass']

        # TODO : the setup of mean_speeds should be much better than that. And also, this should be max(sigma*c_r), not max(sigma)max(c_r)
        # we can not multiply those two as mean_speeds can be of a lesser dimension than sigma
        
        
        # p['setup']['dsmc']['max_proba'] = 2*np.max(mean_speeds)*np.max(p['setup']['cross_sections_matrix'])*np.ones(p['setup']['dsmc']['grid'].current.shape)
        mean_speeds_init_proba = np.full((len(species['names'])), 1e-15)
        c = 0
        for i, s in enumerate(species['names']):
            
            if(s in p['setup']['particles_initialization']): # in particles init - priority to init to
                params_s = p['setup']['particles_initialization'][s]
                mean_speeds_init_proba[i] = get_mean_speed(params_s)
            elif(p['use_fluxes'] and s in fluxes['names']): # in fluxes
                mean_speeds_init_proba[i] = mean_speeds[c]
                c+=1
            else : # unchanged, but in this case, the particles is 'useless' because it's is not initialized and not injected.
                   # unless of course it it will becreated trough reactions
                   # so maybe try a better init here
                pass

        p['setup']['dsmc']['max_proba'] = init_max_proba(radii, mean_speeds_init_proba, dsmc['cells_number']) # grid use a 1D-structure
        p['setup']['dsmc']['cell_volume'] = dsmc['cell_volume']
        p['setup']['dsmc']['particles_weight'] = dsmc['particles_weight']
        p['setup']['dsmc']['mean_numbers_per_cell'] = dsmc['mean_numbers_per_cell']
            # fluxes
    if(p['use_fluxes']):
        p['setup']['inject'] = {}
        p['setup']['inject']['species'] = fluxes['names']
        p['setup']['inject']['in_wall'] = fluxes['in_wall']
        p['setup']['inject']['in_vect'] = fluxes['in_vect']
        p['setup']['inject']['debits'] = fluxes['debits']
        p['setup']['inject']['vel_stds'] = fluxes['vel_stds']
        p['setup']['inject']['drifts'] = fluxes['drifts']
        # reactions
    if(p['use_reactions']):
        p['setup']['reactions'] = {}        
        p['setup']['reactions']['boundaries'] = ld.advection.reactions.parse(p['reactions']['boundaries'])
        # poisson
    if(p['use_poisson']):
        p['setup']['mesh'] = ps.mesh.polygonal(poisson['mesh_resolution'],  np.flip(np.array(points), axis = 0), out_vertices_list = None) # polygonal recieves points in counter-clock order
        p['setup']['potential_field'], p['setup']['electric_field'] = ps.solver(p['setup']['mesh'], poisson['boundary_conditions'], poisson['charge_density'])

        # integration
    if(p['use_poisson']):
        p['setup']['schemes'], p['setup']['update_functions'], p['setup']['args_update_functions'] = \
            integration_setup(p['simulation']['integration'], p['setup']['species'], masses, charges, p['setup']['electric_field'], p['setup']['potential_field'])
    else :
        p['setup']['schemes'], p['setup']['update_functions'], p['setup']['args_update_functions'] = \
            integration_setup(p['simulation']['integration'], p['setup']['species'], masses, charges)

    # since it is a inplace function, we dont need to return anything.
    
# ------------------ utils -------------------------- #

def integration_setup(integration_params, species_to_int, masses, charges, electric_field = None, potential_field = None):
    nb_species = len(species_to_int)
    default_scheme, default_fn, default_args = None, None, None
    if('default' in integration_params):
        default_scheme = ld.utils.schemes.scheme_dispatcher(integration_params['default']['scheme'])
        default_fn = integration_params['default']['fn']
        default_args = integration_params['default']['users_args']
    schemes = [default_scheme]*nb_species
    fn = [default_fn]*nb_species
    args = [dict(default_args) for k in range(nb_species)]

    for key, val in integration_params.items():
        if(key == 'default'):
            continue
        idx = species_to_int[key]

        schemes[idx] = ld.utils.schemes.scheme_dispatcher(val['scheme'])
        fn[idx] = val['fn']
        args[idx] = val['users_args']

    for k in range(len(args)):
        args[k]['mass'] = masses[k]
        args[k]['charge'] = charges[k]

        for k in range(nb_species):
            args[k]['electric_field'] = None 
            if(electric_field is not None and potential_field is not None):
                args[k]['electric_field'] = electric_field
            # args[k]['potential_field'] = potential_field

    return schemes, fn, args

def reflection_functions_setup(reflection_params, species_to_int):
    nb_species = len(species_to_int)
    default_reflect_fn = None, None, None

    if('default' in reflection_params):
        default_reflect_fn = ld.advection.boundaries.reflection_functions_dispatcher(reflection_params['default']['reflect_fn'])
        default_user_generic_args = reflection_params['default']['generic_args'] # list
        default_user_defined_args = reflection_params['default']['args'] # dict

    reflect_fns = [default_reflect_fn]*nb_species
    user_generic_args_list = [default_user_generic_args]*nb_species
    user_defined_args_list = [dict(default_user_defined_args) for k in range(nb_species)]

    for key, val in reflection_params.items():
        if(key == 'default'):
            continue
        idx = species_to_int[key]

        reflect_fns[idx] = val['reflect_fn']
        user_generic_args_list[idx] = val['generic_args']
        user_defined_args_list[idx] = val['args']

    return reflect_fns, user_generic_args_list, user_defined_args_list
    
def particles_initialization_setup(init_params:dict, keys_to_int:dict, masses:np.ndarray, quantities:np.ndarray):
    species = init_params['species']
    types = init_params['types']
    params = init_params['params'] # dict of list - keys are the species
    # dict linking species to [type, quantity, params]
    init_params_setup = {}

    for k, t in enumerate(types):
        # maxwellian init required the mass of the species (when giving the temp)
        if(t == 'maxwellian'): params[species[k]] = [params[species[k]][0], masses[keys_to_int[species[k]]]] 
        elif(t == 'uniform'): pass # nothing to add here (mass is not required here)

        init_params_setup[species[k]] = {
            'params' : params[species[k]],
            'type' : t,
            'quantity' : int(quantities[keys_to_int[species[k]]])
        }

    return init_params_setup

def get_mean_speed(params):
    t = params['type']
    if(t =='maxwellian'):
        return ld.utils.physics.maxwellian_mean_speed(params['params'][0],params['params'][1]) # temperature, mass
    elif(t  =='uniform'):
        return (np.abs(params['params'][0])+np.abs(params['params'][1]))*0.5 # (min + max)*0.5
    else:
        print(f'Type {t} not recognized. Should be : maxwellian, uniform.')
        return None

def init_max_proba(radii, mean_speeds, nb_cells_in_grid):
    # radii and v_mean are of size Ns(number of species)
    # grid_shape is of size Nc (the number of cells)
    shape_pmax = [radii.shape[0], radii.shape[0], nb_cells_in_grid]
    pmax = np.ones(tuple(shape_pmax), dtype = float)
    for i in range(pmax.shape[0]):
        for j in range(i+1):
            if(i==j):
                pmax[i,j] *= 2*mean_speeds[i] * np.pi * 4*radii[i]**2  # we'll take the max proba straight away ?
            else:
                pmax[i,j] *= np.abs(mean_speeds[i]-mean_speeds[j]) * np.pi * (radii[i]+radii[1])**2   # we'll take the max proba straight away ?z
    return pmax.max(axis = (0,1))

def get_cross_sections(radii):
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
