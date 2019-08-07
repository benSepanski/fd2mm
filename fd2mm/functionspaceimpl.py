import numpy as np

from firedrake import VectorFunctionSpace, FunctionSpace, project
from firedrake.functionspaceimpl import WithGeometry

from fd2mm.analog import Analog
from fd2mm.finat_element import FinatElementAnalog
from fd2mm.mesh import MeshTopologyAnalog, MeshGeometryAnalog, \
    MeshAnalogWithBdy

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
    def __init__(self, function_space, function_space_analog, mesh_analog,
                 near_bdy=None, on_bdy=None):
        # FIXME docs
        # FIXME use near bdy
        """

            :arg near_bdy: Same as for :class:`MeshAnalogNearBdy`
            :arg on_bdy: Same as for :class:`MeshAnalogOnBdy`
        """
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

        # Check near_bdy and on_bdy
        assert near_bdy is None or on_bdy is None
        # If one is not *None*, store it in bdy_id
        bdy_id = on_bdy
        if near_bdy is not None:
            bdy_id = near_bdy
        if bdy_id is None:
            # can't convert whole mesh if mesh analog only has bdy
            assert not isinstance(mesh_analog, MeshAnalogWithBdy)
        else:
            # otherwise make sure converting whole mesh, or at least
            # portion with given boundary
            assert not isinstance(mesh_analog, MeshAnalogWithBdy) or \
                mesh_analog.contains_bdy(bdy_id)

        # Initialize as Analog
        super(WithGeometryAnalog, self).__init__(function_space)

        self._shared_data = \
            FunctionSpaceDataAnalog(mesh_analog,
                                    function_space_analog.finat_element_a,
                                    function_space._shared_data)

        self._mesh_a = mesh_analog
        self._topology_a = function_space_analog

        # If mesh has higher order than the function space, we need this
        # to project into
        self._intermediate_fntn = None

    def __getattr__(self, attr):
        return getattr(self._topology_a, attr)

    def mesh_a(self):
        return self._mesh_a

    def meshmode_mesh(self):
        return self._shared_data.meshmode_mesh()

    def _reordering_array(self, firedrake_to_meshmode):
        if firedrake_to_meshmode:
            return self._shared_data.firedrake_to_meshmode()
        else:
            return self._shared_data.meshmode_to_firedrake()

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

        # {{{ handle higher order meshes
        mesh_order = \
            self.analog().mesh().coordinates.function_space().finat_element.degree

        order = self.analog().finat_element.degree
        if order < mesh_order:
            # Create fntn if haven't already, then project into it
            if self._intermediate_fntn is None:
                mesh = self.analog().mesh()
                family = self.analog().ufl_element().family()
                shape = self.analog().shape
                if shape:
                    if len(shape) > 1:
                        raise NotImplementedError(
                            "Can only convert scalars and vectors")
                    V = VectorFunctionSpace(mesh, family, mesh_order, dim=shape[0])
                else:
                    V = FunctionSpace(mesh, family, mesh_order)
                self._intermediate_fntn = project(function, V)
            else:
                project(function, self._intermediate_function)
            function = self._intermediate_function

        # }}}

        nodes = function.dat.data

        # handle vector function spaces differently
        if len(nodes.shape) > 1:
            new_nodes = [self.reorder_nodes(nodes.T[i], True) for i in
                         range(nodes.shape[1])]
            nodes = np.array(new_nodes)
        else:
            nodes = self.reorder_nodes(nodes, True)

        return nodes
