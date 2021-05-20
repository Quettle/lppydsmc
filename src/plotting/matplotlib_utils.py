import matplotlib.pyplot as plt
import numpy as np


# ------------------- System, particles and grid ------------------ #

def plot_system(arr, segments, radius, gs, ss, offset):
    fig, ax = plt.subplots()
    plot_grid(ax, gs, ss, offset)
    plot_boundaries(ax, segments)
    if(arr is not None):
        plot_particles(ax, arr, radius)
    ax.axis('equal')
    return fig, ax

def plot_boundaries(ax, segments, color = 'k'):
    # segments is a 2D nd array, where segmment[int] = segment = [x1, y1, x2, y2]
    for k in range(segments.shape[0]):
        ax.plot(segments[k, [0,2]], segments[k, [1,3]], color = color)

def plot_particles(ax, arr, r, arrows = False, factor = 0.01, color = 'b', arrow_color = 'r', line = False): 
    # Arr is a 2D nd array, where arr[int] = particle = [x,y,vx,vy,vz]
    if(type(color) == str):
        ax.scatter(arr[:,0], arr[:,1], s = np.pi*r*r, color = color)
    else:
        ax.scatter(arr[:,0], arr[:,1], s = np.pi*r*r, c = color)
    if(arrows):
        for k in range(arr.shape[0]):
            ax.arrow(arr[k,0], arr[k,1], dx = factor*1.5*r*arr[k,2], dy = factor*1.5*r*arr[k,3], 
                    width = factor*0.1*r*0.3, head_width = 3*factor*r*0.3, head_length = 1.5*factor*r*0.3,  \
                        length_includes_head = True, color = arrow_color)
    if(line):
        for k in range(arr.shape[0]):
            ax.plot([arr[k,0]+10*factor*1.5*r*arr[k,2], arr[k,0]-10*factor*1.5*r*arr[k,2]],[arr[k,1]+10*factor*1.5*r*arr[k,3], arr[k,1]-10*factor*1.5*r*arr[k,3]], '--', color = 'g')

def plot_grid(ax, gs, ss, offset = np.array([0,0]), color = 'g'):
    # gs : grid shape
    # ss : system shape
    segments = _get_segments(gs, ss, offset)
    plot_boundaries(ax, segments, color = color)

def _get_segments(gs, ss, offset):
    segments = []
    lx, ly = ss[0], ss[1]
    dx, dy = ss[0]/gs[0], ss[1]/gs[1]
    segments.append([0, 0, 0, ly])
    segments.append([lx, ly, 0, ly])
    segments.append([0, 0, lx, 0])
    segments.append([lx, 0, lx, ly])

    for i in range(1, gs[0]):
        segments.append([i*dx, 0, i*dx, ly])
    for j in range(1, gs[1]):
        segments.append([0, j*dy, lx, j*dy])

    return np.array(segments)+np.concatenate((offset, offset), axis = 0)

