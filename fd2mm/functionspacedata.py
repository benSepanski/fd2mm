import numpy as np
from decorator import decorator
from fd2mm.analog import Analog
from fd2mm.finat_element import FinatElementAnalog
from fd2mm.utils import reorder_nodes


@decorator
def cached(f, mesh_analog, key, *args, **kwargs):
    """
    Exactly :func:`firedrake.functionspacedata.cached`, but
    caching on mesh Geometry instead

    :arg f: The function to cache.
    :arg mesh_analog: The mesh_analog to cache on (should have a
        ``_shared_data_cache`` object).
    :arg key: The key to the cache.
    :args args: Additional arguments to ``f``.
    :kwargs kwargs:  Additional keyword arguments to ``f``."""
    assert hasattr(mesh_analog, "_shared_data_cache")
    cache = mesh_analog._shared_data_cache[f.__name__]
    try:
        return cache[key]
    except KeyError:
        result = f(mesh_analog, key, *args, **kwargs)
        cache[key] = result
        return result


@cached
def higher_degree_finat_element_analog(mesh_analog, finat_element_analog):
    """
        Pick the finat element between coordinates finat element and
        :meth:`finat_element_analog` which is of higher degree. If
        the coordinates finat element is higher degree, make a new
        finat element which matches
        :attr:`finat_element_analog().analog().family`
    """
    from fd2mm.mesh import MeshTopologyAnalog
    if isinstance(mesh_analog, MeshTopologyAnalog):
        return finat_element_analog

    mesh_fe_analog = mesh_analog.coordinates_a.function_space_a().finat_element_a
    fe_analog = finat_element_analog
    if mesh_fe_analog.analog().degree > fe_analog.analog().degree:
        raise NotImplementedError("BEN, this is implemented, just untested")
        if isinstance(mesh_fe_analog, type(fe_analog)):
            fe_analog = mesh_fe_analog
        else:
            fe_analog = type(fe_analog)(fe_analog.cell,
                                        mesh_fe_analog.degree)
    return fe_analog


def get_nodes(mesh_analog, key):
    """
        Computes nodes using given mesh analog and finat element analog

        :arg mesh_analog: A :class:`MeshAnalog`
        :arg key: A:class:`FinatElementAnalog`
    """
    if not isinstance(key, FinatElementAnalog):
        raise TypeError(":arg:`key` must be of class FinatElementAnalog")

    nodes_fe_analog = higher_degree_finat_element_analog(mesh_analog, key)

    ambient_dim = mesh_analog.analog().geometric_dimension()
    nelements = mesh_analog.nelements()
    nunit_nodes = nodes_fe_analog.nunit_nodes()

    nodes = np.zeros((ambient_dim, nelements, nunit_nodes))

    coordinates = mesh_analog.analog().coordinates
    coord_fspace = coordinates.function_space()
    coords_fe_analog = mesh_analog.coordinates_a.function_space_a().finat_element_a

    for i, indices in enumerate(coord_fspace.cell_node_list):
        elt_coords = np.real(coordinates.dat.data[indices].T)
        # handle 1D-case
        if len(elt_coords.shape) == 1:
            elt_coords = elt_coords.reshape(1, elt_coords.shape[0])

        # NOTE : Here, we are in effect 'creating' nodes for CG spaces,
        #        since come nodes that were shared along boundaries are now
        #        treated as independent
        #
        #        In particular, this node numbering may be different
        #        than firedrake's!

        # If have same number of vertices as nodes, they are the same
        if nunit_nodes == len(indices):
            nodes[:, i, :] = elt_coords[:, :]
        else:
            nodes[:, i, :] = \
                coords_fe_analog.map_points(elt_coords, nodes_fe_analog.unit_nodes())

    return nodes


@cached
def get_meshmode_mesh(mesh_analog, finat_element_analog):

    vertex_indices = mesh_analog.vertex_indices()
    vertices = mesh_analog.vertices()
    nodes = get_nodes(mesh_analog, finat_element_analog)

    # Get the fe analog used for nodes
    fe_analog = higher_degree_finat_element_analog(mesh_analog, finat_element_analog)

    from meshmode.mesh import SimplexElementGroup
    # Nb: topological_dimension() is a method from the firedrake mesh
    group = SimplexElementGroup(
        fe_analog.analog().degree,
        vertex_indices,
        nodes,
        dim=mesh_analog.cell_dimension(),
        unit_nodes=fe_analog.unit_nodes())

    from meshmode.mesh.processing import flip_simplex_element_group
    group = flip_simplex_element_group(vertices, group,
                                       mesh_analog.orientations() < 0)

    from meshmode.mesh import Mesh
    return Mesh(vertices, [group],
                boundary_tags=mesh_analog.bdy_tags(),
                nodal_adjacency=mesh_analog.nodal_adjacency(),
                facial_adjacency_groups=mesh_analog.facial_adjacency_groups())


@cached
def reordering_array(mesh_analog, key, fspace_data):
    """
    :arg key: A tuple (finat_element_anlog, firedrake_to_meshmode)
    where *firedrake_to_meshmode* is a *bool*, *True* indicating
    firedrake->meshmode reordering, *False* meshmode->firedrake

    Returns a *np.array* that can reorder the data by composition,
    see :meth:`fd2mm.function_space.FunctionSpaceAnalog.reorder_nodes`
    """
    finat_element_analog, firedrake_to_meshmode = key

    cell_node_list = fspace_data.entity_node_lists[mesh_analog.analog().cell_set]
    num_fd_nodes = fspace_data.node_set.size
    group = get_meshmode_mesh(mesh_analog, finat_element_analog).groups[0]
    num_mm_nodes = group.nnodes

    if firedrake_to_meshmode:
        nnodes = num_fd_nodes
    else:
        nnodes = num_mm_nodes
    order = np.arange(nnodes)

    # Put into cell-node list if firedrake-to meshmode (so can apply
    # flip-mat)
    if firedrake_to_meshmode:
        new_order = order[cell_node_list]
    # else just need to reshape new_order so that can apply flip-mat
    else:
        nunit_nodes = group.nunit_nodes
        new_order = order.reshape(
            (order.shape[0]//nunit_nodes, nunit_nodes) + order.shape[1:])

    flip_mat = finat_element_analog.flip_matrix()
    reorder_nodes(mesh_analog.orientations(), new_order, flip_mat,
                  unflip=firedrake_to_meshmode)
    new_order = new_order.flatten()

    # Resize new_order if going meshmode->firedrake and meshmode
    # has duplicate nodes (e.g if used a CG fspace)
    #
    # this is done VERY LAZILY (NOT GOOD)
    if not firedrake_to_meshmode and num_fd_nodes != num_mm_nodes:
        newnew_order = np.zeros(num_fd_nodes, dtype=np.int32)
        pyt_ndx = 0
        for nodes in cell_node_list:
            for fd_index in nodes:
                newnew_order[fd_index] = new_order[pyt_ndx]
                pyt_ndx += 1

        new_order = newnew_order

    # }}}

    return new_order


class FunctionSpaceDataAnalog(Analog):
    """
        This is not *quite* the usual thought of a :class:`Analog`.

        It is an analog in the sense that a mesh & finat element
        define a lot of the data of a function space, so we use
        this object to store data based on the mesh and finat element.
        For us, however, we need the geometry (so that we know the

        :arg fspace_data: A :class:`firedrake.functionspacedata.FunctionSpaceData`
                          matching :attr:`mesh_analog.topological_a.analog()`
                          and :attr:`finat_element_analog.analog()`

        Note that :arg:`mesh_analog` and :arg:`finat_element_analog`
        are used for analog-checking
    """

    def __init__(self, mesh_analog, finat_element_analog, fspace_data):
        if mesh_analog.topological_a == mesh_analog:
            raise TypeError(":arg:`mesh_analog` is a MeshTopologyAnalog,"
                            " must be a MeshGeometryAnalog")
        analog = (mesh_analog.analog(), finat_element_analog.analog())
        super(FunctionSpaceDataAnalog, self).__init__(analog)

        self._mesh_analog = mesh_analog
        self._finat_element_analog = finat_element_analog
        self._fspace_data = fspace_data

    def meshmode_mesh(self):
        return get_meshmode_mesh(self._mesh_analog, self._finat_element_analog)

    def firedrake_to_meshmode(self):
        return reordering_array(self._mesh_analog,
                                (self._finat_element_analog, True),
                                self._fspace_data)

    def meshmode_to_firedrake(self):
        return reordering_array(self._mesh_analog,
                                (self._finat_element_analog, False),
                                self._fspace_data)
