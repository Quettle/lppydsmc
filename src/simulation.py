def position_in_grid(pos, system_size, grid_size):
    """ This functions returns the simples version of 'how to compute a given position in a grid when given the size of the system and the size of the grid'.
       /!\ It supposes that the particle is in the system. /!\
        For example : *int(-0.5)* is equal to  *0* thus it will consider such a position to be in the grid whereas it is obviously not. 


    Args:
        pos (tuple or list of float): the position to be 'converted'
        system_size (tuple or list of float): the nd-size of the system
        grid_size (tuple or list of int): the nd-size of the grid (should be int)

    Returns:
        pos_in_grid: a tuple forming the position in grid of the given input position.
    """
    pos_in_grid = tuple([int(a*c/b) for a,b,c in zip(pos, system_size, grid_size)])
    return pos_in_grid

def advection_handler(dt, list_particles, walls, update_fn, args, handle_wall_collision):
    pass

def handle_wall_collision(part, wall):
    pass

