from dolfin.functions.functionspace import VectorFunctionSpace, FunctionSpace

def electric_field(mesh, mesh_dict, phi_dict, physics_consts_dict):
    
    # V = FunctionSpace(mesh, 'P', 1)
    W = VectorFunctionSpace(mesh = mesh, family = 'P', degree = 1) # , dim = 2 # dim corresponds to the dimension of mesh and is optionnal (in this case it is deduces from mesh)
    
    L_mot = mesh_dict['L_mot']
    l_mot = mesh_dict['l_mot']
    L_vacuum = mesh_dict['L_vacuum']
    l_vacuum = mesh_dict['l_vacuum']
    L_1 = mesh_dict['L_1']
    l_1 = mesh_dict['l_1']
    L_2 = mesh_dict['L_2']
    l_2 = mesh_dict['l_2']
    L_3 = mesh_dict['L_3']
    l_3 = mesh_dict['l_3']
    delta_vert_12 = mesh_dict['delta_vert_12']
    delta_vert_23 = mesh_dict['delta_vert_23']
    mesh_resolution = mesh_dict['mesh_resolution']
    refine_mesh = mesh_dict['refine_mesh']

    h_grid = l_1 + l_2 + l_3 + delta_vert_12 + delta_vert_23

    class Boundary_top_mot(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and x[1] == l_mot/2
        
    class Boundary_bord_mot(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and x[1] < l_mot/2  and x[1] > - l_mot/2 + tol
        
    class Boundary_electrode1(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and x[1] <= - l_mot/2 + tol and x[1] >= - l_mot/2 - l_1 - tol
        
    class Boundary_inter_electrode(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            test_1 = on_boundary and x[1] < - l_mot/2 - l_1 - tol and x[1] > - l_mot/2 - l_1 - delta_vert_12 + tol
            test_2 = on_boundary and x[1] < - l_mot/2 - l_1 - delta_vert_12 - l_2 - tol and x[1] > - l_mot/2 - l_1 - delta_vert_12  - l_2 - delta_vert_23 + tol
            return test_1 or test_2
        
    class Boundary_electrode2(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and x[1] <= - l_mot/2 - l_1 - delta_vert_12 + tol and x[1] >= - l_mot/2 - \
                l_1 - delta_vert_12 - l_2 - tol and abs(x[0])<=L_mot/2
        
    class Boundary_electrode3(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and x[1] <= - l_mot/2 - l_1 - delta_vert_12 - l_2 - delta_vert_23 + tol and x[1] >= - l_mot/2 - \
                l_1 - delta_vert_12 - l_2 - delta_vert_23 -l_3 - tol and abs(x[0])<=L_mot/2
        
    class Boundary_sup_vacuum(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and (( x[1]< -l_mot/2 - h_grid and x[1] >= -l_mot/2 - \
                h_grid - l_vacuum/2) or (x[1]== -l_mot/2 - h_grid and abs(x[0])>L_mot/2))
        
    class Boundary_inf_vacuum(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and x[1]< -l_mot/2 - h_grid - l_vacuum/2

    top_mot = Boundary_top_mot()
    bord_mot = Boundary_bord_mot()
    electrode1 = Boundary_electrode1()
    inter_electrode = Boundary_inter_electrode()
    electrode2 = Boundary_electrode2()
    electrode3 = Boundary_electrode3()
    sup_vacuum = Boundary_sup_vacuum()
    inf_vacuum = Boundary_inf_vacuum()

    list_Phi = [phi_dict['Phi_top_mot'],phi_dict['Phi_bord_mot'],phi_dict['Phi_electrode1'], \
            phi_dict['Phi_inter_electrode'], phi_dict['Phi_electrode2'], phi_dict['Phi_electrode3'], \
            phi_dict['Phi_sup_vacuum'],phi_dict['Phi_inf_vacuum']]
    list_edges=[top_mot, bord_mot, electrode1, inter_electrode, electrode2, electrode3, sup_vacuum, inf_vacuum]
    bc=[]
    
    for i in range(len(list_edges)):
        if list_Phi[i]!='N':
            bc.append(DirichletBC(V, Constant(list_Phi[i]), list_edges[i]))


    u = TrialFunction(V)
    v = TestFunction(V)
    
    rhoelec=physics_consts_dict['rhoelec']
    l_rho=physics_consts_dict['l_rho']
    
    class Second_membre(UserExpression):
        def eval(self, value, x):
            ylim = -(0.5*l_mot + l_1 + delta_vert_12 + l_2 + delta_vert_23 + l_3)
            
            if l_rho!=0:
                
                if x[1] >= ylim + l_rho:
                    value[0] = rhoelec/physics_consts_dict['PERMITTIVITY']
                    
                elif x[1] >= ylim:
                    a = rhoelec*((-ylim - l_rho)**(-.5) - (-ylim)**(-.5))**(-1)
                    b = - a*(-ylim)**(-.5)
                    value[0] =  a * (-x[1])**(-.5) + b 
                    value[0] /= physics_consts_dict['PERMITTIVITY']
                    
                else:
                    value[0]=0
                    
            else:
                value[0]=rhoelec/physics_consts_dict['PERMITTIVITY']
    
    f = Second_membre(degree=2)

    #f = Constant(physics_consts_dict['rhoelec']/physics_consts_dict['PERMITTIVITY']) 
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    Phi = Function(V)

    solve(a == L, Phi, bc)

    E = project(-grad(Phi), W)

    return Phi,E,f
