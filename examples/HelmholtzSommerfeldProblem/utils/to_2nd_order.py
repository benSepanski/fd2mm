import numpy.linalg as la
from firedrake import VectorFunctionSpace, project
from firedrake.mesh import MeshGeometry


def to_2nd_order(mesh, circle_bdy_id=None, rad=1.0):

    # make new coordinates function
    V = VectorFunctionSpace(mesh, 'CG', 2)
    new_coordinates = project(mesh.coordinates, V)

    fac_nr_to_nodes = V.finat_element.entity_closure_dofs()[
        mesh.topological_dimension() - 1]

    # If we have a circle, move any nodes on the circle bdy
    # onto the circle. Note circle MUST be centered at origin
    if circle_bdy_id is not None:
        for marker, icells, ifacs in zip(mesh.exterior_facets.markers,
                                         mesh.exterior_facets.facet_cell,
                                         mesh.exterior_facets.local_facet_number):
            # if on the circle
            if marker == circle_bdy_id:
                for icell, ifac in zip(icells, ifacs):
                    cell_nodes = V.cell_node_list[icell]
                    cell_nodes_on_fac = cell_nodes[fac_nr_to_nodes[ifac]]

                    #Force all cell nodes to have given radius :arg:`rad`
                    for node in cell_nodes_on_fac:
                        scale = rad / la.norm(new_coordinates.dat.data[node])
                        new_coordinates.dat.data[node] *= scale

    # Make a new mesh with given coordinates
    c_fspace = mesh.coordinates.function_space()
    elt = c_fspace.ufl_element()
    new_elt = type(elt)(elt.family(), elt.cell(), 2)
    
    new_mesh = MeshGeometry.__new__(MeshGeometry, new_elt)
    new_mesh.__init__(new_coordinates.topological)

    return new_mesh
