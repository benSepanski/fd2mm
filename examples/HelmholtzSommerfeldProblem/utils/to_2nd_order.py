import numpy.linalg as la
from firedrake import VectorFunctionSpace, project, Mesh


def to_2nd_order(mesh, circle_bdy_id=None, rad=1.0):

    # make new coordinates function
    V = VectorFunctionSpace(mesh, 'CG', 2)
    new_coordinates = project(mesh.coordinates, V)

    # If we have a circle, move any nodes on the circle bdy
    # onto the circle. Note circle MUST be centered at origin
    if circle_bdy_id is not None:
        nodes_on_circle = V.boundary_nodes(circle_bdy_id, 'geometric')
        #Force all cell nodes to have given radius :arg:`rad`
        for node in nodes_on_circle:
            scale = rad / la.norm(new_coordinates.dat.data[node])
            new_coordinates.dat.data[node] *= scale

    # Make a new mesh with given coordinates
    return Mesh(new_coordinates)
