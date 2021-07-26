import numpy as np

"""
Help for defining the boundaries conditions for the thruster problem. Note : the dimensions are fixes and given in *dimensions_default*.
"""

dp = 0.001

dimensions_default = {
            'w_in' : 5*dp,
            'l_in' : 3*dp,
            'w_1' : 3*dp,
            'l_1' : dp,
            'l_int' : dp,
            'w_2' : dp,
            'l_2' : 10*dp,
            'w_out' : 5*dp,
            'l_out' : dp,
            'offsets' : np.array([0,0]) 
        }

charge_density_default = {
            'value' : '0',  # must be a string too
            'degree' : 0,
            'kwargs' : {}
            }

def thruster(dimensions = dimensions_default, mesh_resolution = 100, potential_electrode_1 = '30', potential_electrode_2 = '300', potential_electrode_3 = None, charge_density = charge_density_default):
    
    from .mesh import polygonal
    from .solver import solver
    from ..systems.helper import thruster_points, thruster_three_grids_points

    tol = 1e-14

    # useful lenghts to define boundary conditions
    x_in = dimensions['offsets'][0]
    x_electrode_1 = x_in + dimensions['l_in']
    x_inter_electrodes_area = x_electrode_1 + dimensions['l_1']
    x_electrode_2 = x_inter_electrodes_area + dimensions['l_int']
    if(potential_electrode_3 is not None):
        x_inter_electrodes_area_2 = x_electrode_2 + dimensions['l_2']
        x_electrode_3 = x_inter_electrodes_area_2 + dimensions['l_int_2']
        x_out = x_electrode_3 + dimensions['l_3']

    else:
        x_out = x_electrode_2 + dimensions['l_2']

    potential_boundary_conditions = {
        'inflow_area' : '0.0',
        'inflow_area_sides': 'Neumann',
        'electrode_1' : str(potential_electrode_1),
        'inter_electrode_area':'Neumann',
        'electrode_2': str(potential_electrode_2),
        'inter_electrode_area_2':'Neumann',
        'electrode_3': str(potential_electrode_3)
    }

    # boundary conditions
    boundary_conditions = {
                    'inflow_area' : {
                        'type' : 'Dirichlet',
                        'solution' : potential_boundary_conditions['inflow_area'], 
                        'solution_kwargs' : {},
                        'degree' : 0, # because constant
                        'boundary' : 'on_boundary && near(x[0], x_in, tol)', 
                        'boundary_kwargs' : {
                            'tol' : tol,
                            'x_in' : x_in
                        }               
                    },
                    'inflow_area_sides' : {
                        'type' : 'Neumann'
                    },
                    'electrode_1' : {
                        'type' : 'Dirichlet',
                        'solution' : potential_boundary_conditions['electrode_1'], 
                        'solution_kwargs' : {},
                        'degree' : 0,
                        'boundary' : 'on_boundary && x[0] > x_electrode_1 - tol && x[0] < x_inter_electrodes_area + tol',
                        'boundary_kwargs' : {
                            'tol' : tol,
                            'x_electrode_1' : x_electrode_1,
                            'x_inter_electrodes_area' : x_inter_electrodes_area
                        }
                    },
                    'inter_electrode_area' : {
                        'type' : 'Neumann'
                    },
                    'electrode_2' : {
                        'type' : 'Dirichlet',
                        'solution' : potential_boundary_conditions['electrode_2'], 
                        'solution_kwargs' : {},
                        'degree' : 0,
                        'boundary' : 'on_boundary && x[0] > x_electrode_2 - tol && x[0] < x_out + tol',
                        'boundary_kwargs' : {
                            'tol' : tol,
                            'x_electrode_2' : x_electrode_2,
                            'x_out' : x_out
                        }
                    }
                }

    if(potential_electrode_3 is not None):
        boundary_conditions['electrode_3'] ={
                    'type' : 'Dirichlet',
                    'solution' : potential_boundary_conditions['electrode_3'], 
                    'solution_kwargs' : {},
                    'degree' : 0,
                    'boundary' : 'on_boundary && x[0] > x_electrode_3 - tol && x[0] < x_out + tol',
                    'boundary_kwargs' : {
                        'tol' : tol,
                        'x_electrode_3' : x_electrode_3,
                        'x_out' : x_out
                    }
                }
        boundary_conditions['inter_electrode_area_2'] = {
                    'type' : 'Neumann'
                }
        in_vertices = np.flip(thruster_three_grids_points(**dimensions), axis = 0)

    else:

        # mesh creation
        in_vertices = np.flip(thruster_points(**dimensions), axis = 0)
    mesh = polygonal(resolution = mesh_resolution, in_vertices = in_vertices)

    # potential_fields / electric_field
    potential_field, electric_field = solver(mesh, boundary_conditions, charge_density) 

    return potential_field, electric_field
