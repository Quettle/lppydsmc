import dolfin
import mshr
from fenics import * 

import numpy as np

def polygonal(resolution, in_vertices, out_vertices_list = None):
    """Creates a mesh, of resolution *resolution*, from  *in_vertices* and *out_vertices* and as a polygon. 

    Args:
        resolution (int): resolutions of the mesh
        in_vertices (list of numpy arrays size 2 - 1D): the vertices corresponding to form the polygon in counter-clock order
        out_vertices (list of numpy arrays size 2 - 1D, optional): the vertices to form the out polygone in counter-clock order, default : None.

    Returns:
        mshr.mesh : the mesh corresponding to the given vertices.
    """
    in_vertices = [Point(in_vertices[k,0],in_vertices[k,1]) for k in range(in_vertices.shape[0])] 

    domain = mshr.Polygon(in_vertices) # https://bitbucket.org/fenics-project/mshr/wiki/API/Polygon
                                     # Create polygon defined by the given vertices. Vertices must be in counter-clockwise order and free of self-intersections.
    
    if(out_vertices_list is not None):
        for out_vertices in out_vertices_list:
            out_vertices = [Point(out_vertices[k,0],out_vertices[k,1]) for k in range(out_vertices.shape[0])]
            domain -= mshr.Polygon(out_vertices)
    
    mesh=mshr.generate_mesh(domain, resolution)

    # if(refine_mesh):
    #     d = mesh.topology().dim()
        
    #     class To_refine(SubDomain):
    #         def inside(self, x, on_boundary):
    #             return x[1]<=0 and x[1]>= -l_mot/2-h_grid-l_vacuum/4

    #     to_refine = To_refine()
    #     marker = MeshFunction("bool", mesh, d, False)
    #     to_refine.mark(marker, True)
    #     mesh = refine(mesh,marker)

    return mesh