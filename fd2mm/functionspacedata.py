import numpy as np
from decorator import decorator
from firedrake.functionspacedata import FunctionSpaceData
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory
import numpy.linalg as la

from fd2mm.analog import Analog
from fd2mm.finat_element import FinatElementAnalog


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


def reorder_nodes(orient, nodes, flip_matrix, unflip=False):
    """
        :arg orient: An array of shape (nelements) of orientations,
                     >0 for positive, <0 for negative
        :arg nodes: a (nelements, nunit_nodes) or (dim, nelements, nunit_nodes)
                    shaped array of nodes
        :arg flip_matrix: The matrix used to flip each negatively-oriented
                          element
        :arg unflip: If *True*, use transpose of :arg:`flip_matrix` to
                     flip negatively-oriented elements

        flips :arg:`nodes`
    """
    # reorder nodes (Code adapted from
    # meshmode.mesh.processing.flip_simplex_element_group)

    # ( round to int bc applying on integers)
    flip_mat = np.rint(flip_matrix)
    if unflip:
        flip_mat = flip_mat.T

    # flipping twice should be identity
    assert la.norm(
        np.dot(flip_mat, flip_mat)
        - np.eye(len(flip_mat))) < 1e-13

    # }}}

    # {{{ flip nodes that need to be flipped, note that this point we act
    #     like we are in a DG space

    # if a vector function space, nodes array is shaped differently
    if len(nodes.shape) > 2:
        nodes[orient < 0] = np.einsum(
            "ij,ejk->eik",
            flip_mat, nodes[orient < 0])
        # Reshape to [nodes][vector dims]
        nodes = nodes.reshape(
            nodes.shape[0] * nodes.shape[1], nodes.shape[2])
        # pytential wants [vector dims][nodes] not [nodes][vector dims]
        nodes = nodes.T.copy()
    else:
        nodes[orient < 0] = np.einsum(
            "ij,ej->ei",
            flip_mat, nodes[orient < 0])
        # convert from [element][unit_nodes] to
        # global node number
        nodes = nodes.flatten()


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
    assert isinstance(finat_element_analog, FinatElementAnalog)

    cell_node_list = fspace_data.entity_node_lists[mesh_analog.analog().cell_set]
    if mesh_analog.icell_to_fd is not None:
        cell_node_list = cell_node_list[mesh_analog.icell_to_fd]

    num_fd_nodes = fspace_data.node_set.size

    nelements = cell_node_list.shape[0]
    nunit_nodes = cell_node_list.shape[1]
    num_mm_nodes = nelements * nunit_nodes

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


@cached
def get_factory(mesh_analog, degree):
    return InterpolatoryQuadratureSimplexGroupFactory(degree)


@cached
def get_discretization(mesh_analog, key):
    finat_element_analog, cl_ctx = key
    assert isinstance(finat_element_analog, FinatElementAnalog)

    degree = finat_element_analog.analog().degree
    discretization = Discretization(cl_ctx,
                                    mesh_analog.meshmode_mesh(),
                                    get_factory(mesh_analog, degree))

    return discretization


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

    # FIXME: Give two finat elts
    def __init__(self, cl_ctx, mesh_analog, finat_element_analog):
        if mesh_analog.topological_a == mesh_analog:
            raise TypeError(":arg:`mesh_analog` is a MeshTopologyAnalog,"
                            " must be a MeshGeometryAnalog")
        analog = (mesh_analog.analog(), finat_element_analog.analog())
        super(FunctionSpaceDataAnalog, self).__init__(analog)

        self._fspace_data = FunctionSpaceData(mesh_analog.analog(),
                                              finat_element_analog.analog())

        self._cl_ctx = cl_ctx
        self._mesh_analog = mesh_analog
        self._finat_element_analog = finat_element_analog
        self._discretization = None

    def firedrake_to_meshmode(self):
        return reordering_array(self._mesh_analog,
                                (self._finat_element_analog, True),
                                self._fspace_data)

    def meshmode_to_firedrake(self):
        return reordering_array(self._mesh_analog,
                                (self._finat_element_analog, False),
                                self._fspace_data)

    def discretization(self):
        return get_discretization(self._mesh_analog,
                                  (self._finat_element_analog, self._cl_ctx))

    def factory(self):
        degree = self._finat_element_analog.analog().degree
        return get_factory(self._mesh_analog, degree)
