"""Used to raise *UserWarning*s"""
from warnings import warn
import pyopencl as cl
import numpy as np

from firedrake import SpatialCoordinate, Function, \
    VectorFunctionSpace
from firedrake.functionspaceimpl import WithGeometry
from finat.fiat_elements import Lagrange

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory
from meshmode.discretization.connection import make_face_restriction

from pytential import bind
from pytential.qbx import QBXLayerPotentialSource
from pytential.target import PointsTarget

from fd2mm import FunctionAnalog


"""
These are helper functions so that someone who is new to
pytential/meshmode only has to write down an operator and bind
"""


class SourceConnection:
    """
        firedrake->meshmode

        Note that you should NOT set :arg:`with_refinement` to be
        *True* unless the source has co-dimension 1
    """
    def __init__(self, cl_ctx, fspace_analog, bdy_id=None):

        degree = self._fspace_analog.analog().finat_element.degree
        mesh = self._fspace_analog.meshmode_mesh()

        self.factory = InterpolatoryQuadratureSimplexGroupFactory(degree)
        discr = Discretization(cl_ctx, mesh, self.factory)
        connection = None

        if bdy_id is not None:
            connection = make_face_restriction(self.discr, self.factory, bdy_id)
            discr = connection.to_discr

        self._discr = discr
        self._connection = connection

    def get_qbx(self, **kwargs):
        """
            Return a :class:`QBXLayerPotentialSource` to bind
            to an operator
        """
        qbx = QBXLayerPotentialSource(self._discr, **kwargs)
        return qbx

    def __call__(self, queue, function_analog, bdy_id=None):
        """
            Convert this function to a discretization on the given device
        """
        field = cl.array.to_device(queue, function_analog.as_field())

        if self._connection is not None:
            field = self._connection(queue, field)

        return field


class TargetConnection:
    """
        meshmode->firedrake
    """
    def __init__(self, function_space):
        self._function_space = function_space
        self._function_space_a = None
        self._target_indices = None
        self._target = None

    def set_function_space_analog(self, function_space_analog):
        """
            Set a function space analog
        """
        self._function_space_a = function_space_analog

    def get_function_space(self):
        """
            Return this object's function space
        """
        return self._function_space

    def get_target(self):
        """
            Return a target to bind to an operator
        """
        return self._target

    def set_bdy_as_target(self, bdy_id, method):
        """
            :arg bdy_id: An iterable of subdomain ids as could be
                              accepted as the :arg:`subdomain` in
                              a :class:`DirichletBC` from :mod:`firedrake`.
            :arg method: A method for determining bdy nodes as
                         in :class:`DirichletBC', either 'geometric'
                         or 'topological'
        """
        mesh = self._function_space.mesh()

        # if just passed an int, convert to an iterable of ints
        # so that just one case to deal with
        if isinstance(bdy_id, int):
            bdy_id = [bdy_id]
        target_markers = set(bdy_id)

        # Check that bdy ids are valid
        if not target_markers <= set(mesh.exterior_facets.unique_markers):
            warn("The following bdy ids are not exterior facet ids: %s" %
                 (target_markers - set(mesh.exterior_facets.unique_markers)))

        if not target_markers & set(mesh.exterior_facets.unique_markers):
            raise ValueError("No bdy ids are exterior facet ids")

        self._target_indices = set()
        for marker in target_markers:
            self._target_indices |= set(
                self._function_space.bdy_nodes(marker, method))
        self._target_indices = np.array(list(self._target_indices), dtype=np.int32)

        # Get coordinates of nodes
        coords = SpatialCoordinate(mesh)
        function_space_dim = VectorFunctionSpace(
            mesh,
            self._function_space.ufl_element().family(),
            degree=self._function_space.ufl_element().degree())

        coords = Function(function_space_dim).interpolate(coords)
        coords = np.real(coords.dat.data)

        target_pts = coords[self._target_indices]
        # change from [nnodes][ambient_dim] to [ambient_dim][nnodes]
        target_pts = np.transpose(target_pts).copy()
        self._target = PointsTarget(target_pts)

    def set_function_space_as_target(cl_ctx, self):
        """
            PRECONDITION: Have set a function space analog for the function space
                          used, else raises ValueError

            Sets whole function space as target in converted meshmode
            form
        """
        if self._function_space_a is None:
            raise ValueError("No converter set")

        if isinstance(self._function_space.finat_element, Lagrange):
            warn("Careful! :mod:`meshmode` uses all DG elements."
                 " You are trying to convert DG -> CG"
                 " (pytential->fd) [DANGEROUS--ONLY DO IF YOU KNOW RESULT"
                 " WILL BE CONTINUOUS]")

        # Set unrefined qbx as target_qbx
        degree = self._function_space_a.analog().finat_element.degree
        mesh = self._function_space_a.meshmode_mesh()
        factory = InterpolatoryQuadratureSimplexGroupFactory(degree)

        self._target = Discretization(cl_ctx, mesh, factory)

    def __call__(self, queue, result, result_function_a):
        """
            :arg result_function_a: Either a FunctionAnalog or a Function.
                                    If a Function, the target MUST be
                                    set to a boundary

            Converts the result of a pytential operator bound to this
            target (or really any correctly-sized array)
            to a firedrake function on this object's
            :meth:`get_function_space`

            PRECONDITION: target must be set
        """
        if self._target is None:
            raise ValueError("No target set")

        if isinstance(self._target, PointsTarget):
            if isinstance(result_function_a, FunctionAnalog):
                result_function_a = result_function_a.analog()
            if len(result.shape) > 1:
                for i in range(result.shape[0]):
                    result_function_a.dat.data[self._target_indices, i] = result[i]
            else:
                result_function_a.dat.data[self._target_indices] = result
        else:
            result_function_a.set_from_field(result)

        return result_function_a


class OpConnection:
    """
        operator evaluation and binding
    """

    def __init__(self, source_connection, target_connection, op):
        """
            A source connection, target connection, and operator
        """
        self._source_connection = source_connection
        self._target_connection = target_connection

        qbx = self._source_connection.get_qbx()
        target = self._target_connection.get_target()

        self._bound_op = bind((qbx, target), op)

    def __call__(self, queue, result_function_a, **kwargs):
        """
            Evaluates the operator for the given function.

            :arg queue: a :mod:`pyopencl` queue to use (usually
                made from the cl_ctx passed to this object
                during construction)
            :arg result_function_a: As for :meth:`TargetConnection.__call__`
            :arg out_function_space: TODO
            :arg **kwargs: Arguments to pass to op. All :mod:`firedrake`
                :class:`Functions` are converted to pytential
        """
        new_kwargs = {}
        for key in kwargs:
            if isinstance(kwargs[key], Function):
                # Convert function to array with pytential ordering
                new_kwargs[key] = self._source_connection(queue, kwargs[key])
            else:
                new_kwargs[key] = kwargs[key]

        # Perform operation and take result off queue
        result = self._bound_op(queue, **new_kwargs)

        # handle multi-dimensional vs 1-dimensional results differently
        # to take array off of device
        if isinstance(result, np.ndarray):
            result = np.array([arr.get(queue=queue) for arr in result])
        else:
            result = result.get(queue=queue)

        self._target_connection(queue, result, result_function_a)


def fd_bind(cl_ctx, fspace_analog, op, source=None, target=None,
            source_only_near_bdy=False, source_only_on_bdy=False,
            with_refinement=False):
    """
        :arg cl_ctx: A cl context
        :arg fspace_analog: A function space analog
        :arg op: The operation
        :arg sources: either
            - A FunctionSpace, which will be the source
            - A pair (FunctionSpace, bdy_id) which will be the source
              (where bdy_id is the bdy which will be the source,
               *None* for the whole mesh)

        :arg targets: either
            - A FunctionSpace, which will be the target
            - A pair (FunctionSpace, bdy_id) which will be the target
              (where bdy_id is the bdy which will be the target,
               *None* for the whole mesh)

        At least one of the following two arguments must be *False*
        :arg source_only_near_bdy: If *True*, allow conversion of
                                   source function space only
                                   near the give source bdy id, if
                                   one is given.
        :arg source_only_on_bdy: If *True*, allow conversion of
                                   source function space only
                                   on the give source bdy id, if
                                   one is given.

        :arg with_refinement: If *True*, use refined qbx for source, this
                              is highly recommended
    """
    assert not (source_only_near_bdy and source_only_on_bdy)

    # Source and target will now be (fspace, bdy_id or *None*)
    if isinstance(source, WithGeometry):
        source = (source, None)
    if isinstance(target, WithGeometry):
        target = (target, None)

    source_connection = SourceConnection(cl_ctx, fspace_analog,
                                         bdy_id=source[1])

    target_connection = TargetConnection(target[0])
    if target[1] is None:
        target_connection.set_function_space_analog(fspace_analog)
        target_connection.set_function_space_as_target(cl_ctx)
    else:
        target_connection.set_bdy_as_target(target[1], 'geometric')

    return OpConnection(source_connection, target_connection, op)
