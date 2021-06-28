from fenics import *

VACUUM_PERMITTIVITY = 8.8541878128e-12 # F.m-1

def solver(mesh, boundary_conditions, charge_density):
    """ Solves the poisson's equation.

    Args:
        mesh (mshr.mesh): mesh of the system
        boundary_conditions (dict): Dictionarry containing the different boundary conditions. See below for an example.
        charge_density (dict): Dictionary giving the formula to compute the charge density (depending on x and y) and the degree of interpolation required.

        Example :
            In practice, you need not specify any Neumann conditions as they are taken as default.
            Expressions in string format have to be in C++.
            ```
            boundary_conditions = { 
                'top' : {
                    'type' : 'Dirichlet',
                    'solution' : '1 + x[0]*x[0] + 2*x[1]*x[1]', # 2D, x[1] = 'x', x[2] = 'y' for example.
                    'solution_kwargs' : {},
                    'degree' : 2,
                    'boundary' : 'on_boundary && near(x[0], 1, tol)', # must return a boolean
                                                                      # on_boundary corresponds to a boolean fenics returns, because it knows in practice if *x* is on the boundary or not.
                                                                      # *near(x[0], 1, tol)* is equivalent to *abs(x[0]-1)<tol*
                    'boundary_kwargs' : {
                        'tol' : 1e-14
                    }               
                },
                'bottom' : {
                    'type' : 'Dirichlet',
                    'solution' : 'x[1] <= 0.5 + tol ? k_0 : k_1',
                    'solution_kwargs' : {
                        'k_0' : 1.0,
                        'k_1' : 0.01,
                        'tol' : 1e-14
                    },
                    'degree' : 0,
                    'boundary' : 'x[1] <= 0.5 + tol ? true : false',
                    'boundary_kwargs' : {
                        'tol' : 1e-14
                    }
                }
            }

            charge_density = {
                'value' : 'max_density-x[0]*(max_density-min_density)/lx',  # must be a string too
                'degree' : 2,
                'kwargs' : {
                    'max_density' : 2e17,
                    'min_density' : 1e17,
                    'lx' : 1.0
                } 
            }
    Returns:
        (): returns the potential (scalar) and electrical (vector) fields.
    """
    # please refer to to understand the method in place here : https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1003.html#ch-fundamentals
    # please also refer to : https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html for more information
    # boundaries_fn must return a boolean value - True is the point x lies on the boundary 

    # Defining the finite element function space
    V = FunctionSpace(mesh, 'Lagrange', 1,  1) # mesh, family, degree
    # Defining boundary conditions 
    bc_list = []
    for key, value in boundary_conditions.items():
        # creating boundary condition (only dirichlet), by default Fenics takes Neumann. Robin not (yet) taken into account.
        if(value['type'] == 'Dirichlet'):
            # solution on the boundary
            u_d = Expression(value['solution'], degree=value['degree'], **value['solution_kwargs']) # degree should be at least 1 - better too for more accuracy
            # sub_domain = CompiledSubDomain(value['boundary'], **value['boundary_kwargs'])
            # temporary solution :
            cmd  = value['boundary']
            for key, val in value['boundary_kwargs'].items():
                cmd = cmd.replace(key, str(val))
            bc_list.append(DirichletBC(V, u_d, cmd)) #  , **value['boundary_kwargs'] - boundary defines a subdomain of the main domain - and it is a string

    # Defining source term
    charge_density = Expression('{}/VACUUM_PERMITTIVITY'.format(charge_density['value']), degree=charge_density['degree'], VACUUM_PERMITTIVITY = VACUUM_PERMITTIVITY, **charge_density['kwargs']) # Note : optimization can be made in case the source_term is constant

    # Defining the variational problem
    u = TrialFunction(V) # Trial function
    v = TestFunction(V) # Test function
    a = dot(grad(u), grad(v))*dx # bilinear form - canonical notation
    L = charge_density*v*dx # linear form - canonical notation

    #  solving poisson's equation (electric potential)
    u = Function(V)
    solve(a == L, u, bc_list)

    # computing electric field
    W = VectorFunctionSpace(mesh = mesh, family = 'Lagrange', degree = 1) # , dim = 2 # dim corresponds to the dimension of mesh and is optionnal (in this case it is deduces from mesh)
    electic_field = project(-grad(u), W)

    return u, electic_field
