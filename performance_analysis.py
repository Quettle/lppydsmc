# system
from src.system_creator import SystemCreator

# Grid
from src.utils import Grid, pos_in_grid

# Particles
from src.utils import Particle

# injection 
from src.utils import inject

# advection
from src.utils import advect
from src.utils import euler_explicit, leap_frog

# collisions
from src.utils import handler_wall_collision, handler_wall_collision_point, make_collisions_vectorized, make_collisions_out_walls

# utils 
from src.utils import gaussian, maxwellian_flux, maxwellian_mean_speed, get_mass_part

# plotting 
from src.plotting import plot_boundaries, plot_particles, plot_grid

# collisions between particles
from src.utils import handler_particles_collisions # candidates, index_choosen_couples, probability, is_colliding, reflect, 

# other imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from icecream import ic
import pandas as pd
import seaborn as sns

np.random.seed(1111)


def thruster(w_in, l_in, w1, l1, l_int, w2, l2, w_out, l_out, offsets = np.array([0,0])):
    # hypothesis : w_int = w_in
    # returns an array with the walls for the thruster
    # not optimized but we prioritize the clarity here
    def rectangle(w,l, offset=np.array([0,0])):
        # top left point is p1 and then its trigo rotation
        p1 = np.array([0,0])+offset
        p2 = np.array([w,0])+offset
        p3 = np.array([w,-l])+offset
        p4 = np.array([0,-l])+offset
        return p1,p2,p3,p4

    p1, p2, p3, p20 = rectangle(w_in,l_in)
    p19, p4, p5, p18 = rectangle(w1,l1, offset = np.array([0.5*(w_in-w1),-l_in]))
    p17, p6, p7, p16 = rectangle(w_in,l_int, offset = np.array([0, -l1-l_in]))
    p15, p8, p9, p14 = rectangle(w2, l2, offset = np.array([0.5*(w_in-w2),-l_in-l1-l_int]))
    p13, p10, p11, p12 = rectangle(w_out, l_out, offset = np.array([0.5*(w_in-w_out),-l_in-l1-l_int-l2]))
    points = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20])
    segments = np.concatenate((points[1:],points[:19]), axis = 1)
    segments = np.concatenate((segments, np.expand_dims(np.concatenate((p20,p1)),axis = 0)), axis = 0)
    # sorting is realized when the array is created per the SystemCreator. No need to worry at this point.
    return segments

def mean_free_path(cross_section, density):
    return 1/(cross_section*density)

def mean_free_time(mfp, v_mean):
    return mfp/v_mean

def plot(arr, segments, radius, gs, ss, offset):
    fig, ax = plt.subplots()
    plot_grid(ax, gs, ss, offset)
    plot_boundaries(ax, segments)
    if(arr is not None):
        plot_particles(ax, arr, radius)
    plt.axis('equal')
    plt.show()
    
def convert_to_grid_datatype(positions, new, old = 0):
    index_container = np.zeros((new-old))
    index_in_container = np.arange(old, new)
    indexes = np.stack((index_container, index_in_container), axis = 1)
    return np.concatenate((positions, indexes), axis = 1).astype(int)

iterations = 1000

# System :
dz = 0.001
idx_out_walls = [1,2]
# idx_out_walls = [0,10] # 2nd and 3rd walls are out walls.

    # tube
segments = 0.001*np.array([[0,0,10,0], [0,0,0,1], [10,0,10,1], [0,1,10,1]]) # tube
system = SystemCreator(segments)

    # Thruster
# dp = 0.001
# segments = thruster(w_in = 5*dp, l_in = 3*dp, w1 = 3*dp, l1 = dp, l_int = dp, w2 = dp, l2 = 5*dp, w_out = 5*dp, l_out = dp, offsets = np.array([0,0]))
# system = SystemCreator(segments)
    # cylinder
# res = 4
# circle = [[1.5+0.5*np.cos(k*np.pi/res), 1+0.5*np.sin(k*np.pi/res), 1.5+0.5*np.cos((k+1)*np.pi/res), 1+0.5*np.sin((k+1)*np.pi/res)] for k in range(2*res)]
# segments = 0.001*np.array([[0,0,3,0], [0,0,0,2], [3,0,3,2], [0,2,3,2]]+circle)
# system = SystemCreator(segments)

offsets = system.get_offsets()
system_shape = system.system_shape()
a = system.get_dir_vects()
segments = system.get_segments()

# grid :
mean_number_per_cell = 1000
max_number_per_cell = 10*mean_number_per_cell
resolutions = np.array((10,1), dtype = int)
grid = Grid(resolutions, max_number_per_cell)
volume_cell = dz * system_shape[0]/resolutions[0] * system_shape[1]/resolutions[1]

# Particles - 1 type 
density = 3.2e19 # m-3
n_simu = mean_number_per_cell*np.sum(resolutions)
n_real = dz * system_shape[0] * system_shape[1] * density
mr = n_real/n_simu# macro particules ratio = number of particles in the real system / number of macro part in the simulated system
density_dsmc = density/mr
temperature = 300 # K

part_type = 'I'
charge, mass, radius = 0, get_mass_part(53, 53, 74), 2e-10
size_array = 2*max_number_per_cell*np.sum(resolutions)
v_mean = maxwellian_mean_speed(temperature, mass)
container = Particle(part_type, charge, mass, radius, size_array)
cross_section = container.get_params()[3]

# mean free path and time
mfp = mean_free_path(cross_section, density)
typical_lenght = 0.001
mft = mean_free_time(typical_lenght, v_mean = v_mean)

# Injection params
in_wall = np.array([0,0,0,0.001], dtype = float)
in_vect = np.array([1,0], dtype = float)
debit = maxwellian_flux(density_dsmc, v_mean)*np.linalg.norm(in_wall[:2]-in_wall[2:])*dz
vel_std = gaussian(temperature, mass)
dt = 2e-6 # in sec, should be a fraction of the mean free time

# advection
def f(arr, dt):
    return np.zeros(shape = (arr.shape[0], 3))
args = []
scheme = euler_explicit


#### -------------- Simulating 


remains = 0
df = pd.DataFrame(columns = ['x','y','vx','vy','vz'])
saving_period = 1
# dsmc saving
averages = np.full(shape = grid.current.shape, fill_value = mean_number_per_cell)
remains_per_cell = np.zeros(shape = grid.current.shape, dtype = float)
# pmax = np.full(shape = grid.current.shape, fill_value = 2*vel_std*cross_section)
pmax = 2*vel_std*cross_section
arr_nb_colls = np.zeros((iterations, resolutions[0], resolutions[1]))
for it in tqdm(range(iterations)): # tqdm
    n1 = container.get_current()
                   
    # injecting particles
    new, remains = inject(in_wall, in_vect, debit, vel_std, radius, dt, remains)
    container.add_multiple(new)
                   
    n2 = container.get_current()-n1
    
    # PHASE : ADVECTING
        # MOVING PARTICLES
    arr = container.get_particles()
    
    if(it%saving_period==0):
        df = df.append(pd.DataFrame(data=arr, index=[it]*arr.shape[0], columns = ['x','y','vx','vy','vz']))
    
    advect(arr, f, dt, args, scheme) # advect is inplace
    
        # HANDLING BOUNDARIES 
    # positions_save = np.copy(arr) # for later
    count = np.full(fill_value = True, shape = arr.shape[0])
    idxes_out = []
    c = 0
    while(np.sum(count, where = count == True) > 0):
        c+=1
        ct, cp = handler_wall_collision_point(arr[count], segments, a) # handler_wall_collision(arr[count], segments, a, radius)
        count, idxes_out_ = make_collisions_out_walls(arr, a, ct, cp, idx_out_walls, count) # idxes_out : indexes of the particles (in arr) that got out of the system
        idxes_out.append(idxes_out_)
        
    idxes_out = np.concatenate(idxes_out)
    container.delete_multiple(idxes_out)
    arr = container.get_particles()
    
    # PHASE : COLLISIONS
        # UPDATING GRID - HARD RESET
    grid.reset()
    positions = pos_in_grid(arr[:,:2], resolutions, offsets, system_shape)
    particles = convert_to_grid_datatype(positions, new = positions.shape[0])
    grid.add_multiple(particles)

        # DSMC
        # TODO: make parallel
    #arr_save = np.copy(arr)
    currents = grid.get_currents()
    remains_per_cell, nb_colls = handler_particles_collisions([arr], grid.get_grid(), currents, dt, averages, pmax, cross_section, volume_cell, mr, remains_per_cell)
    arr_nb_colls[it,:,:] = nb_colls
    # if(np.array_equal(arr_save, container.get_particles()) and np.sum(nb_colls)>0):
    #    print('Arrays are equals but there were collisions.')
    
    # PLOTTING (OPTIONAL)
    # if(it%100==0):
        #plot(arr, segments, radius, resolutions, system_shape, offsets)
    #    print('{:^10}|{:^10}|{:^10}|{:^10}|{:^10}'.format(it, n1, n2, idxes_out.shape[0], c))

