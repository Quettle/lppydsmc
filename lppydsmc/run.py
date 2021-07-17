import lppydsmc as ld
import numpy as np
import inspect
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def run(path_to_cfg, save):
    p = ld.config.cfg_reader.read(path_to_cfg) # parameters

    setup(p) # modify p in place

    if(save): # savnig the new params to a file so the user can go and debug it or refer to all the simulations params when needed.
        from configobj import ConfigObj
        pp_dict = convert_objects(p.dict())
        pp = ConfigObj(pp_dict)
        pp.filename = '{}/{}.ini'.format(p['directory'], p['name'])
        pp.write()
        
    # TODO :
    # - add plot functions and params
    # - add monitoring
    # - add more complexe system (with parts of the system that we dont take) - example of the cylinder.

    # maybe some verbose and saving of the params 

    # SIMULATION
    simulate(p['use_fluxes'], p['use_dsmc'], p['use_reactions'], monitoring = None, **p['setup'])

    return p

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

def simulate(use_fluxes, use_dsmc, use_reactions, monitoring, **kwargs):
    species = kwargs['species'] # dict
    system = kwargs['system']
    containers = kwargs['containers'] # dict
    cross_sections_matrix = kwargs['cross_sections_matrix']

    # simulation
    time_step = kwargs['time_step'] # float
    iterations = kwargs['iterations'] # int

    # advect
    update_functions = kwargs['update_functions']
    args_update_functions = kwargs['args_update_functions']
    schemes = kwargs['schemes']
    if(use_fluxes):
        injected_species = kwargs['inject']['species'] # list is enough here I think
        in_wall = kwargs['inject']['in_wall']
        in_vect = kwargs['inject']['in_vect']
        debits = kwargs['inject']['debits']
        vel_stds = kwargs['inject']['vel_stds']
        drifts = kwargs['inject']['drifts']
        
        remains = np.zeros((len(injected_species)))

    if(use_dsmc):
        grid = kwargs['dsmc']['grid']
        resolutions = kwargs['dsmc']['resolutions']
        max_proba = kwargs['dsmc']['max_proba']
        cell_volume = kwargs['dsmc']['cell_volume']
        particles_weight = kwargs['dsmc']['particles_weight']

        # useful values
        remains_per_cell = np.zeros(shape = grid.current.shape, dtype = float)
        averages = np.full(shape = grid.current.shape, fill_value = np.sum(kwargs['dsmc']['mean_numbers_per_cell']))

    if(use_reactions):
        boundaries_reactions = kwargs['reactions']['boundaries']
   

        # useful 
    masses = np.array([container.mass() for specie, container in containers.items()])
    time = 0.

    # SIMULATING HERE
    fig, ax = plt.subplots(constrained_layout = True)
    for iteration in tqdm(range(iterations)):
        if(use_fluxes): 
            inject(injected_species, containers, in_wall, in_vect, debits, vel_stds, time_step, remains, drifts)
        
        advect([container.get_array() for specie, container in containers.items()], time, time_step, update_functions, args_update_functions, schemes)

        idxes_gsi = reflect_out_particles(containers, system, iteration, monitoring)

        if(use_reactions):
            recombine(idxes_gsi, containers, boundaries_reactions, masses, species, monitoring)

        if(use_dsmc):
            try :
                dsmc(containers, grid, resolutions, system.get_offsets(), system.get_shape(), averages, iteration, time_step, max_proba, cell_volume, particles_weight, cross_sections_matrix, remains_per_cell, monitoring)
            except Exception:
                # fig, ax = plt.subplots(constrained_layout = True)
                ax.clear()
                plot_system(ax, containers, system)
                plt.show(fig)
                raise Exception

        time+=time_step

        if(iteration%1==0):
            ax.clear()
            plot_system(ax, containers, system)
            plt.savefig(f'debugs/{iteration}.png')
# ---------------- Plotting ---------------- #
def plot_system(ax, containers, system):
    for segment in system.get_segments():
        ax.plot(segment[[0,2]], segment[[1,3]], color = 'k')

    for specie, container in containers.items():
        arr = container.get_array()
        ax.scatter(arr[:,0], arr[:,1], label = specie)
    ax.legend(loc='best')


# ------------------- Inject -------------------- #
def inject(species, containers, in_wall, in_vect, debits, vel_stds, dt, remains, drifts): # in place for remains because numpy array.
    # some species may not be injected while others will /!\
    for i, k in enumerate(species): # idx, key (example : 0, 'I')
        new, remains[i] = ld.injection.maxwellian(in_wall, in_vect, debits[i], vel_stds[i], dt, remains[i], drifts[i])
        containers[k].add_multiple(new)

# ----------------- Advect ----------------------- #
def advect(arrays, time, time_step, update_functions, args_update_functions, schemes): # in place
    for k in range(len(arrays)):
        ld.advection.advect(arrays[k], update_functions[k], time_step, time, args_update_functions[k], schemes[k]) # advect is inplace 

# --------------- Handle boundaries -------------- #

def reflect_out_particles(containers, system, iteration, monitoring = None): # will also delete particles that got out of the system.
    a = system.get_dir_vects()
    segments = system.get_segments()
    idx_out_segments = system.get_idx_out_segments()

    list_counts = []
    for k, (specie, container) in enumerate(containers.items()):
        # initializing local variable
        arr = container.get_array()
        container = containers[specie]
        
        count = np.full(fill_value = True, shape = arr.shape[0])
        idxes_out = []
        collided = []
        c = 0
        
        while(np.count_nonzero(count) > 0): # np.sum(count, where = count == True) > 0):
            c+=1
            ct, cp, cos_alpha = ld.advection.wall_collision.handler_wall_collision_point(arr[count], segments, a) # handler_wall_collision(arr[count], segments, a, radius)
            count, idxes_out_, cos_alpha = ld.advection.wall_collision.make_collisions_out_walls(arr, a, ct, cp, idx_out_segments, count, cos_alpha) # idxes_out : indexes of the particles (in arr) that got out of the system
            idxes_out.append(idxes_out_)

            # the first one that is received is the number of particles colliding with walls.
            if(c == 1):
                if(monitoring is not None):
                    monitoring['collisions_with_walls'] += np.count_nonzero(count) # np.sum(count, where = count == True)
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
            if(monitoring is not None):
                monitoring['df_out_particles'] = monitoring['df_out_particles'].append(pd.DataFrame(data=np.concatenate((out_arr, np.expand_dims([k]*out_arr.shape[0], axis = 1)), axis = 1), index=[iteration]*out_arr.shape[0], columns = ['x','y','vx','vy','vz','species']))
        else:
            list_counts.append(np.array([])) # appending empty list to conserve the correspondance between rank in the global list and species

    return list_counts # should at some point also return cos_alpha 

def recombine(idxes_gsi, containers, reactions, masses, species, monitoring = None): # catalytic boundary recombination - inplace 
    particles_to_add = {}
    arrays = [container.get_array() for specie, container in containers.items()]
    for k, type_part in enumerate(containers):
        if type_part in reactions:
            reacting_particles, particles_to_add_ = ld.advection.reactions.react(np.array(idxes_gsi[k]), arrays = arrays, masses = masses, types_dict = species, reactions = reactions[type_part], p = None)
            # reacting_particles should be deleted in arrays[k]
            # particles_to_add should be added the the asssociated array
            idxes_gsi[k] = reacting_particles # updating the actually reacting particles

            for key, val in particles_to_add_.items():
                if(key in particles_to_add):
                    particles_to_add[key] += val
                else:
                    particles_to_add[key] = val
        else:
            idxes_gsi[k] = np.array([]) 
    # then and only then we delete everything in the update list_counts
    # print(f'Total collision with walls: {collisions_with_walls}')
    # print('DELETE')
    for k, (type_part, container) in enumerate(containers.items()): # here it's only one particle as it is colliding with the wall
        # thus it is easier to delete
        # print('{} - {}'.format(types[k], list_counts[k].shape[0]))
        container.delete_multiple(idxes_gsi[k])
        if(type_part in particles_to_add):
            # print('ADDING - {} - {}'.format(types[k], len(particles_to_add[types[k]])))
            containers[type_part].add_multiple(np.array(particles_to_add[type_part]))

# --------------- dmsc -------------- #

def dsmc(containers, grid, resolutions, system_offsets, system_shape, averages, iteration, time_step, max_proba, cell_volume, particles_weight, cross_sections, remains_per_cell, monitoring = None):
    arrays = [container.get_array() for specie, container in containers.items()]
    
    grid.reset()
    for k in range(len(arrays)):
        arr = arrays[k]
        new_positions = ld.data_structures.grid.default_hashing(ld.data_structures.grid.pos_in_grid(arr[:,:2], resolutions, system_offsets, system_shape), res_y = resolutions[1])  
        parts_in_grid_format = ld.data_structures.grid.convert_to_grid_format(new = new_positions.shape[0], old = 0, container_idx = k)
        grid.add_multiple(new_positions, parts_in_grid_format)
 
    # ----------------------------- PHASE : DSMC COLLISIONS ----------------------------- 
        # TODO: make parallel (1st : note criticals functions in C++)    
    currents = grid.get_currents()
    averages = (iteration*averages+currents)/(iteration+1) # TODO: may be it too violent ? 

    if(monitoring is not None):
        remains_per_cell, nb_colls_, pmax, monitor = ld.collision.handler_particles_collisions(arrays, grid.get_grid(), currents, time_step, \
            averages, max_proba, cross_sections, cell_volume, particles_weight, remains_per_cell, monitoring = True)

        monitoring['nb_colls'] += nb_colls_
        monitoring['pmax'] = pmax
        monitoring['monitor'] = monitor

    else : 
        remains_per_cell = ld.collision.handler_particles_collisions(arrays, grid.get_grid(), currents, time_step, \
            averages, max_proba, cross_sections, cell_volume, particles_weight, remains_per_cell, monitoring = False)

# ------------------- Processing params ------------------- #

def setup(p):
    p['directory'] = Path(p['directory']).resolve()

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

    volume = ld.utils.estimation.estimate_surface(samples = int(1e4), points = points)[0]
    safety_factor = 2
    size_arrays = safety_factor * params_species['densities'] * volume * p['system']['dz'] # volume is approximated using a Monte Carlo method.
    
    cross_sections_matrix = get_cross_sections(params_species['radii']) # hard sphere model

    # TODO : intializing mean speeds is required too, since it will be changed if we need to load something but in a first approximation we need something
    # however we dont know... May be we can ask the user ? 
    # What should we do if there is no injection ? 
    # For now, I think it's safe to say that if use_fluxes != True, then since there is no 'init particles in system' available now, we should simply return an error message
    if(not p['use_fluxes']):
        print('You either need particles in the system (UNAVAILABLE) or have an injection to launch a simulation. Here *use_fluxes* is {}'.format(p['use_fluxes']))
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
            densities = np.array([species[k]['density'] for k in fluxes['names']])

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
    p['setup']['species'] = species['key_to_int']
    p['setup']['cross_sections_matrix'] = cross_sections_matrix
        # simulation
    p['setup']['time_step'] = p['simulation']['time_step'] # float
    p['setup']['iterations'] = p['simulation']['iterations'] # int
        # containers
    types = params_species['types']
    charges = params_species['charges']
    masses = params_species['masses']
    radii = params_species['radii']
    p['setup']['containers'] = {types[k] : ld.data_structures.Particle(types[k], charges[k], masses[k], radii[k], size_arrays[k]) for k in range(params_species['count'])}
        # dsmc
    if(p['use_dsmc']):
        p['setup']['dsmc'] = {}
        p['setup']['dsmc']['grid'] = ld.data_structures.Grid(dsmc['cells_number'], max_size)
        p['setup']['dsmc']['resolutions'] = dsmc['grid']['resolutions']
        # TODO : the setup of mean_speeds should be much better than that. And also, this should be max(sigma*c_r), not max(sigma)max(c_r)
        # we can not multiply those two as mean_speeds can be of a lesser dimension than sigma
        p['setup']['dsmc']['max_proba'] = 2*np.max(mean_speeds)*np.max(p['setup']['cross_sections_matrix'])*np.ones(p['setup']['dsmc']['grid'].current.shape)
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
