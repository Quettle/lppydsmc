import numpy as np
# interpolation
from scipy.interpolate import RectBivariateSpline


# this functions is used to create the vectors saved and then interpolated when simulating
def interpolate(df, offsets, shape, system_size, cell_volume, particles_weight, steady_state = None):
    df['i'] = ((df['x']-offsets[0])*shape[0]/system_size[0]).astype(int) # i is the index of the cell in the x direction
    df['j'] = ((df['y']-offsets[1])*shape[1]/system_size[1]).astype(int) # j is the index of the cell in the y direction
    
    if(steady_state is not None):
        df_particles_steady = df[df.index > steady_state]
        frames_steady_count = df_particles_steady.index.unique().shape[0]
        groups = df_particles_steady[['i','j','vx','vy','vz','species']].groupby(['i','j'])
    else:
        frames_steady_count = df.index.unique().shape[0]
        groups = df[['i','j','vx','vy','vz','species']].groupby(['i','j'])
    
    density = particles_weight*groups['species'].count()/(cell_volume*frames_steady_count)
    mean_vel = groups[['vx','vy','vz']].mean()
    std_vel = groups[['vx','vy','vz']].std()
    
    density_arr = np.zeros(shape)
    dynamic_arr = np.zeros((shape[0],shape[1], 3, 2))

    for (i,j) in density.index: # this way, we have 0 where there are no cells, and a value everywhere else
        density_arr[i,j] = density.loc[i,j]
        dynamic_arr[i,j,:,0] =  mean_vel.loc[i,j]
        dynamic_arr[i,j,:,1] =  std_vel.loc[i,j]
        
    X, Y = np.linspace(offsets[0],offsets[0]+system_size[0], shape[0]), np.linspace(offsets[1],offsets[1]+system_size[1], shape[1])

    return X, Y, density_arr, dynamic_arr

def read_interpolation(path_x, path_y, path_density_arr, path_dynamic_arr = None):
    X = np.load(path_x, allow_pickle=True)
    Y = np.load(path_y, allow_pickle=True)
    density_arr = np.load(path_density_arr, allow_pickle=True)

    density_fn = RectBivariateSpline(X, Y, density_arr) # RectBivariateSpline is much faster on regular grid.
    

    if(path_dynamic_arr is None):
        return density_fn, None
    
    dynamic_arr = np.load(path_dynamic_arr, allow_pickle=True)

    std_fn_vx = RectBivariateSpline(X, Y, dynamic_arr[:,:,0,1])
    std_fn_vy = RectBivariateSpline(X, Y, dynamic_arr[:,:,1,1])
    std_fn_vz = RectBivariateSpline(X, Y, dynamic_arr[:,:,2,1])
    mean_fn_vx = RectBivariateSpline(X, Y, dynamic_arr[:,:,0,0])
    mean_fn_vy = RectBivariateSpline(X, Y, dynamic_arr[:,:,1,0])
    mean_fn_vz = RectBivariateSpline(X, Y, dynamic_arr[:,:,2,0])
    
    def dynamic_fn(X,Y):
        std_vx = np.abs(std_fn_vx.ev(X,Y)) # Take off the np.abs once the negative values have been adresse 
        # (possibly by feeding a better grid to the function) 
        # or maybe select max(0, std_vx)
        # which could yield to a better approximation really
        # I just have to check again why we should have this 'Runge' effect here 
        # this is so weird
        std_vy = np.abs(std_fn_vy.ev(X,Y))
        std_vz = np.abs(std_fn_vz.ev(X,Y))
        mean_vx = mean_fn_vx.ev(X,Y)
        mean_vy = mean_fn_vy.ev(X,Y)
        mean_vz = mean_fn_vz.ev(X,Y)
        VX = np.random.normal(loc=mean_vx, scale=std_vx, size=None)
        VY = np.random.normal(loc=mean_vy, scale=std_vy, size=None)
        VZ = np.random.normal(loc=mean_vz, scale=std_vz, size=None)

        return np.stack((VX,VY,VZ), axis = 1)

    return density_fn.ev, dynamic_fn