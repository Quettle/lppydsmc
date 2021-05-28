import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# my imports
from .matplotlib_utils import plot_boundaries, plot_grid

# --------------------------- basic functions ----------------- #

default_colors = {
    'vx' : 'r',
    'vy' : 'g',
    'vz' : 'b',
    'v' : 'k',
    'x' : 'r',
    'y' : 'g',
    'z' : 'b'
}

def get_color(val):
    try:
        return default_colors[val]
    except KeyError:
        return None

# save fig, with figsize, allows to change the fontsizes of everything. By shrinking the figsize, you automatically augment the fonsize of everything
# this is an easy way of doing things.
def save_fig(fig, path, title = None, dpi = 400, figsize = None):
    if(title is not None):
        fig.suptitle(title)
    if(figsize is not None):
        fig.set_size_inches(*figsize)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def set_axis(ax, x = None, y = None, equal = False):
    ax.autoscale()
    if equal : ax.set_aspect('equal')
    if(x is not None):
        ax.set_xlabel('{} ({})'.format(x, SI_unit(x)))
    if(y is not None):
        ax.set_ylabel('{} ({})'.format(y, SI_unit(y)))

def SI_unit(val):
    if(val in ['particles','density']):
        return '$m^{-3}$'
    elif (val in ['v','vx', 'vy', 'vz', 'sound speed']):
        return '$m.s^{-1}$'
    elif(val ==  'v2'):
        return '$m^2.s^{-2}$'
    elif(val in ['x', 'y', 'z','distance', 'mean free path']):
        return '$m$'
    elif(val in ['t', 'time']):
        return 's'
    elif(val in ['iteration', 'quantity']):
        return '#'
    elif(val in ['mass flow rate']):
        return '$kg.s^{-1}$'
    elif(val in ['temperature', 'T']):
        return 'K'
    elif(val in ['Mach', 'Rate']):
        return ''
    else:
        return ''
    
def state(ax, df, c, segments = None, data_limit = False):
    ax.scatter(df['x'], df['y'], s=0.3, c = c, cmap='seismic') 
    if(segments is not None):
        plot_boundaries(ax, segments)
    if(data_limit):
        min_x, min_y, max_x, max_y = min(df['x']), min(df['y']), max(df['x']), max(df['y'])
        ax.set(xlim=(min_x, max_x), ylim=(min_y, max_y))
    set_axis(ax, 'x', 'y', equal = True)

def hist1d(ax, df, val, bins = 10, density = False, color = 'default'):
    if(color == 'default'):
        color = get_color(val)

    ax.hist(df[val], bins = bins, density = density, color = color)
    set_axis(ax, x = val, y = None)

# TODO: add cbar (color bar)
def hist2d(ax, df, x = 'x', y ='y', stat = 'density', bins = 10, weights = None):
    sns.histplot(df, x=x, y=y, weights = weights, stat = stat, bins = bins, ax = ax)
    set_axis(ax, x = x, y = y, equal = True)

def nb_particles_evolution(ax, df, times = None):
    # we do a group by
    if(times is None):
        ax.plot(df.index.unique(), df['x'].groupby(df.index).agg('count').values)
        set_axis(ax, x = 'iteration', y = 'quantity')
    else:
        ax.plot(times, df['x'].groupby(df.index).agg('count').values)
        set_axis(ax, x = 't', y = 'quantity')

# ----------------------- More complex ones ------------------- #

# very specific functions 
# go back to using basic functions if you want something that fits what you want

def velocity_distribution(df, frames = None, bins = 50, colors = default_colors, density = True, sharex = False, sharey = True):
    fig, axes = plt.subplots(2,2, sharex = sharex, sharey = sharey, dpi = 200, tight_layout = True)

    if(frames is not None):
        df_ = df.loc[np.isin(df.index,frames)]
    else :
        df_ = df
    
    axes[0,0].hist(df_['vx'], bins = bins, density = density, color = colors['vx'])
    axes[0,1].hist(df_['vy'], bins = bins, density = density, color = colors['vy'])
    axes[1,0].hist(df_['vz'], bins = bins, density = density, color = colors['vz'])
    axes[1,1].hist(np.sqrt(df_['vx']**2+df_['vy']**2+df_['vz']**2), bins = bins, density = density, color = colors['v'])

    set_axis(axes[0,0], x = 'vx', y = None)
    set_axis(axes[0,1], x = 'vy', y = None)
    set_axis(axes[1,0], x = 'vz', y = None)
    set_axis(axes[1,1], x = 'v', y = None)

    return fig, axes

def spatial_hist2d(df, frames, val, x_res, y_res, x_step, y_step, bins = 50, color = 'default'):
    if(color == 'default'):
        color = get_color(val)
    
    fig, axes = plt.subplots(y_res,x_res, sharex = True, sharey = True, dpi = 200, tight_layout = True, squeeze=False)
    if(frames is not None):
        df_ = df.loc[np.isin(df.index,frames)]
    else :
        df_ = df
    
    for j in range(axes.shape[0]):
        for i in range(axes.shape[1]):
            axes[j,i].hist(df_.loc[(df_['y']<(j+1)*y_step) & (df_['y']>j*y_step) & (df_['x']<(i+1)*x_step) & (df_['x']>i*x_step)][val], bins = bins, color = color)
            # if(j == axes.shape[0]-1):
            #     set_axis(axes[j,i], x = val, y = None)

    return fig, axes    

# ----------------------- No plotting here ---------------------------- #
def speed_norm(row):
    return np.sqrt(row['vx']*row['vx']+row['vy']*row['vy']+row['vz']*row['vz'])

# DOES NOT WORK
# def spatial_speed(df, frames, x_res, y_res, x_step, y_step):
#     df['mean_speed'] = df.apply(mean_speed, axis = 'columns')
#     arr = np.zeros((x_res, y_res))
#     for j in range(x_res):
#         for i in range(y_res):
#             arr[i,j] = np.mean(df.loc[(df['y']<(j+1)*y_step) & (df['y']>j*y_step) & (df['x']<(i+1)*x_step) & (df['x']>i*x_step)]['mean_speed'].values[:]
#     return arr
