from warnings import warn
import numpy as np

from firedrake import VectorFunctionSpace, FunctionSpace, project
from firedrake.functionspaceimpl import WithGeometry

from fd2mm.analog import Analog
from fd2mm.finat_element import FinatElementAnalog
from fd2mm.mesh import MeshTopologyAnalog, MeshGeometryAnalog

from fd2mm.functionspacedata import FunctionSpaceDataAnalog


class FunctionSpaceAnalog(Analog):
    """
        This is an analog of :class:`firedrake.functionspaceimpl.FunctionSpace`,
        which is not always what results from a call to
        :func:`firedrake.functionspace.FunctionSpace` (it could be a
        WithGeometry, for instance)

        Note that this class is mostly just here to match Firedrake,
        so that we have something to put CoordinatelessFunction analogs
        on
    """
    def __init__(self, function_space, mesh_analog, finat_element_analog):
        # Convert to coordinateless/topological
        function_space = function_space.topological
        mesh_analog = mesh_analog.topological_a

        # {{{ check input
        if not isinstance(finat_element_analog, FinatElementAnalog):
            raise TypeError(":arg:`finat_element_analog` must be of type"
                            " FinatElementAnalog")

        if not isinstance(mesh_analog, MeshTopologyAnalog):
            raise TypeError(":arg:`mesh_analog` must be of type"
                            " MeshGeometryAnalog or MeshTopologyAnalog")

        assert mesh_analog.is_analog(function_space.mesh())
        assert finat_element_analog.is_analog(function_space.finat_element)
        # }}}

        super(FunctionSpaceAnalog, self).__init__(function_space)

        self._mesh_a = mesh_analog
        self.finat_element_a = finat_element_analog

    @property
    def topological_a(self):
        return self

    def mesh_a(self):
        return self._mesh_a


class WithGeometryAnalog(Analog):
    def __init__(self, cl_ctx, function_space, function_space_analog, mesh_analog):
        # FIXME docs
        # FIXME use on bdy
        # {{{ Check input

        if not isinstance(function_space, WithGeometry):
            raise TypeError(":arg:`function_space` must be of type"
                            " :class:`firedrake.functionspaceimpl.WithGeometry")

        if not isinstance(function_space_analog, FunctionSpaceAnalog):
            raise TypeError(":arg:`function_space_analog` must be of type"
                            " FunctionSpaceAnalog")

        if not isinstance(mesh_analog, MeshGeometryAnalog):
            raise TypeError(":arg:`mesh_analog` must be of type"
                            " MeshGeometryAnalog")

        # Make sure analogs are good for given function space
        assert function_space_analog.is_analog(function_space.topological)
        assert mesh_analog.is_analog(function_space.mesh())

        # }}}

        # Initialize as Analog
        super(WithGeometryAnalog, self).__init__(function_space)
        self._topology_a = function_space_analog
        self._mesh_a = mesh_analog
        self._cl_ctx = cl_ctx

        self._shared_data = \
            FunctionSpaceDataAnalog(cl_ctx, mesh_analog,
                                    function_space_analog.finat_element_a)

        mesh_order = mesh_analog.analog().coordinates.\
            function_space().finat_element.degree
        if mesh_order > self.analog().finat_element.degree:
            warn("Careful! When the mesh order is higher than the element"
                 " order conversion MIGHT work, but maybe not..."
                 " To be honest I really don't know.")

        # Used to convert between refernce node sets
        self._resampling_mat_fd2mm = None
        self._resampling_mat_mm2fd = None

    def __getattr__(self, attr):
        return getattr(self._topology_a, attr)

    def mesh_a(self):
        return self._mesh_a

    def _reordering_array(self, firedrake_to_meshmode):
        if firedrake_to_meshmode:
            return self._shared_data.firedrake_to_meshmode()
        else:
            return self._shared_data.meshmode_to_firedrake()

    def factory(self):
        return self._shared_data.factory()

    def discretization(self):
        return self._shared_data.discretization()

    def resampling_mat(self, firedrake_to_meshmode):
        if self._resampling_mat_fd2mm is None:
            element_grp = self.discretization().groups[0]
            self._resampling_mat_fd2mm = \
                self.finat_element_a.make_resampling_matrix(element_grp)

            self._resampling_mat_mm2fd = np.linalg.inv(self._resampling_mat_fd2mm)

        # return the correct resampling matrix
        if firedrake_to_meshmode:
            return self._resampling_mat_fd2mm
        return self._resampling_mat_mm2fd

    def reorder_nodes(self, nodes, firedrake_to_meshmode=True):
        """
        :arg nodes: An array representing function values at each of the
                    dofs, if :arg:`firedrake_to_meshmode` is *True*, should
                    be of shape (ndofs) or (ndofs, xtra_dims).
                    If *False*, should be of shape (ndofs) or (xtra_dims, ndofs)
        :arg firedrake_to_meshmode: *True* iff firedrake->meshmode, *False*
            if reordering meshmode->firedrake
        """
        # {{{ Case where shape is (ndofs,), just apply reordering

        if len(nodes.shape) == 1:
            return nodes[self._reordering_array(firedrake_to_meshmode)]

        # }}}

        # {{{ Else we have (xtra_dims, ndofs) or (ndofs, xtra_dims):

        # Make sure we have (xtra_dims, ndofs) ordering
        if firedrake_to_meshmode:
            nodes = nodes.T

        reordered_nodes = nodes[:, self._reordering_array(firedrake_to_meshmode)]

        # if converting mm->fd, change to (ndofs, xtra_dims)
        if not firedrake_to_meshmode:
            reordered_nodes = reordered_nodes.T

        return reordered_nodes

        # }}}

    def convert_function(self, function):
        from fd2mm.function import FunctionAnalog
        if isinstance(function, FunctionAnalog):
            function = function.analog()

        # FIXME: Check that function can be converted!
        #assert function.function_space() == self.analog()

        nodes = function.dat.data

        # handle vector function spaces differently, hence the shape checks

        # {{{ Reorder the nodes to have positive orientation
        #     (and if a vector, now have meshmode [dims][nnodes]
        #      instead of firedrake [nnodes][dims] shape)

        if len(nodes.shape) > 1:
            new_nodes = [self.reorder_nodes(nodes.T[i], True) for i in
                         range(nodes.shape[1])]
            nodes = np.array(new_nodes)
        else:
            nodes = self.reorder_nodes(nodes, True)

        # }}}

        # {{{ Now convert to pytential reference nodes
        node_view = self.discretization().groups[0].view(nodes)
        # Multiply each row (repping an element) by the resampler
        np.matmul(node_view, self.resampling_mat(True).T, out=node_view)

        # }}}

        return nodes
