"""Used to raise *UserWarning*s"""
from warnings import warn
import pyopencl as cl
import numpy as np

from firedrake import SpatialCoordinate, Function, \
    VectorFunctionSpace
from firedrake.functionspaceimpl import WithGeometry

from pytential import bind
from pytential.target import PointsTarget

from firedrake_to_pytential import FiredrakeMeshmodeConverter
import firedrake_to_pytential.analogs as analogs


class SourceConnection:
    """
        firedrake->meshmode
    """
    def __init__(self, converter, bdy_id=None, with_refinement=None):
        self._converter = converter
        self._bdy_id = bdy_id
        self._with_refinement = with_refinement
        self._queue = None

    def get_qbx(self):
        """
            Return a :class:`QBXLayerPotentialSource` to bind
            to an operator
        """
        qbx = self._converter.get_qbx(bdy_id=self._bdy_id,
                                      with_refinement=self._with_refinement)
        return qbx

    def __call__(self, queue, function):
        """
            Convert this function to a discretization on the given device
        """
        discr = self._converter.convert(queue, function,
                                        bdy_id=self._bdy_id,
                                        with_refinement=self._with_refinement)
        return cl.array.to_device(queue, discr)


class TargetConnection:
    """
        meshmode->firedrake
    """
    def __init__(self, function_space):
        self._function_space = function_space
        self._converter = None
        self._target_indices = None
        self._target = None

    def set_converter(self, converter):
        """
            Set a :class:`FiredrakeMeshmodeConverter`
        """
        self._converter = converter

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

    def set_boundary_as_target(self, boundary_id, method):
        """
            :arg boundary_id: An iterable of subdomain ids as could be
                              accepted as the :arg:`subdomain` in
                              a :class:`DirichletBC` from :mod:`firedrake`.
            :arg method: A method for determining boundary nodes as
                         in :class:`DirichletBC', either 'geometric'
                         or 'topological'
        """
        mesh = self._function_space.mesh()

        # if just passed an int, convert to an iterable of ints
        # so that just one case to deal with
        if isinstance(boundary_id, int):
            boundary_id = [boundary_id]
        target_markers = set(boundary_id)

        # Check that boundary ids are valid
        if not target_markers <= set(mesh.exterior_facets.unique_markers):
            warn("The following boundary ids are not exterior facet ids: %s" %
                 (target_markers - set(mesh.exterior_facets.unique_markers)))

        if not target_markers & set(mesh.exterior_facets.unique_markers):
            raise ValueError("No boundary ids are exterior facet ids")

        self.target_indices = set()
        for marker in target_markers:
            self.target_indices |= set(
                self._function_space.boundary_nodes(marker, method))
        self.target_indices = np.array(list(self.target_indices), dtype=np.int32)

        # Get coordinates of nodes
        xx = SpatialCoordinate(mesh)
        function_space_dim = VectorFunctionSpace(
            mesh,
            self._function_space.ufl_element().family(),
            degree=self._function_space.ufl_element().degree())

        coords = Function(function_space_dim).interpolate(xx)
        coords = np.real(coords.dat.data)

        target_pts = coords[self.target_indices]
        # change from [nnodes][ambient_dim] to [ambient_dim][nnodes]
        target_pts = np.transpose(target_pts).copy()
        self._target = PointsTarget(target_pts)

    def set_function_space_as_target(self):
        """
            PRECONDITION: Have set a converter for the function space
                          used, else raises ValueError

            Sets whole function space as target in converted meshmode
            form
        """
        if self._converter is None:
            raise ValueError("No converter set")

        # Set unrefined qbx as target_qbx
        target_qbx = self._converter.get_qbx(self._function_space)
        self._target = target_qbx.density_discr

    def __call__(self, queue, result, result_function=None):
        """
            Converts the result of a pytential operator bound to this
            target to a firedrake function on this object's
            :meth:`get_function_space`

            PRECONDITION: Have set a converter to use for this conversion
                          if target is set as the whole function space
            PRECONDITION: target must be set
        """
        if self._target is None:
            raise ValueError("No target set")

        if result_function is None:
            result_function = Function(self._function_space)

        if isinstance(self._target, PointsTarget):
            result_function.dat.data[self._target_indices] = result
        else:
            if self._converter is None:
                raise ValueError("No converter set")

            fd_result = self._converter.convert(queue, result,
                                                firedrake_to_meshmode=False)
            result_function.dat.data[:] = fd_result

        return result_function


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
        target = self._target_connection.get_qbx()

        self._bound_op = bind((qbx, target), op)

    def __call__(self, queue, result_function=None, **kwargs):
        """
            Evaluates the operator for the given function.
            Any dof that is not a target point is set to 0.

            :arg queue: a :mod:`pyopencl` queue to use (usually
                made from the cl_ctx passed to this object
                during construction)
            :arg result_function: A function on the function space
                with non-target dofs already set to 0. If not passed in,
                one is constructed. This function will be modified
                and returned.
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
        if isinstance(result, np.ndarray):
            result = np.array([arr.get(queue=queue) for arr in result])
        else:
            result = result.get(queue=queue)

        self._target_connection(queue, result, result_function=result_function)
        return result_function


class ConverterManager:
    """
        This class acts as a manager to generically construct converters
        for :mod:`firedrake` :class:`Function`s and spaces
    """
    def __init__(self, cl_ctx, **kwargs):
        # TODO: Explain more clearly
        """
        :kwargs: These are for the :class:`FiredrakeMeshmodeConverter`,
                 used in the construction of a :mod:`pytential`
                 :class:`QBXLayerPotentialSource`

        """
        self._converters = []
        self._fspace_analogs = []
        self._mesh_analogs = []
        self._finat_element_analogs = []
        self._cell_analogs = []

        self._cl_ctx = cl_ctx
        self._kwargs = kwargs

    def get_converter(self, function_or_space, bdy_id=None, only_near_bdy=None):
        """
            Returns a :class:`FiredrakeMeshmodeConverter`
            which can convert the given function/space
            and boundary id. For arg details, see

            :function:`FiredrakeMeshmodeConverter.can_convert`.

            :arg only_near_bdy: If *True* and :arg:`bdy_id` is not *None*,
                                        then gets any converter that can convert
                                        functions onto the given boundary id.
                                        Otherwise, only return converters
                                        which can convert functions onto
                                        the whole mesh.
        """
        space = function_or_space
        if isinstance(space, Function):
            space = function_or_space.function_space()

        # {{{ Not necessary, just to emphasize default
        if only_near_bdy is None:
            only_near_bdy = False
        # }}}

        # See if already have a converter
        for conv in self._converters:
            if conv.can_convert(space, bdy_id):
                # If don't want to convert near bdy, prevent that
                if not only_near_bdy and not conv.converting_entire_mesh():
                    continue

                return conv

        def check_for_analog(analog_list, obj, near_bdy=None):
            """
                Careful! Not all the is_analog functions
                take *near_bdy* as an arg!
            """
            for pos_analog in analog_list:
                if near_bdy is None:
                    if pos_analog.is_analog(obj):
                        return pos_analog
                else:
                    if pos_analog.is_analog(obj, near_bdy=near_bdy):
                        return pos_analog
            return None

        near_bdy = None  # None if don't want to convert only to near bdy
        if only_near_bdy and bdy_id is not None:
            near_bdy = bdy_id

        # See if have a fspace analog already
        fspace_analog = check_for_analog(self._fspace_analogs, space,
                                         near_bdy=near_bdy)

        # If not, construct one
        if fspace_analog is None:
            # Determine elements
            el_type = None
            from finat.fiat_elements import DiscontinuousLagrange, Lagrange
            if isinstance(space.finat_element, DiscontinuousLagrange):
                el_type = analogs.DGFunctionSpaceAnalog
            elif isinstance(space.finat_element, Lagrange):
                el_type = analogs.CGFunctionSpaceAnalog
            else:
                raise ValueError("Only CG and DG Function spaces are supported!"
                                 " (You are using %s)" % space.finat_element)

            # Check for mesh analog and construct if necessary
            mesh_analog = check_for_analog(self._mesh_analogs, space.mesh(),
                                           near_bdy=near_bdy)
            if mesh_analog is None:
                mesh_analog = analogs.MeshAnalog(space.mesh(), near_bdy=near_bdy)
                self._mesh_analogs.append(mesh_analog)

            # Check for cell analog and construct if necessary
            cell_analog = check_for_analog(self._cell_analogs,
                                           space.finat_element.cell)
            if cell_analog is None:
                cell_analog = analogs.SimplexCellAnalog(space.finat_element.cell)
                self._cell_analogs.append(cell_analog)

            # Check for finat element analog and construct if necessary
            finat_element_analog = check_for_analog(self._finat_element_analogs,
                                                    space.finat_element)
            if finat_element_analog is None:
                finat_element_analog = analogs.FinatElementAnalog(
                    space.finat_element,
                    cell_analog=cell_analog)
                self._finat_element_analogs.append(finat_element_analog)

            # Construct fspace analog
            fspace_analog = el_type(function_space=space,
                                    cell_analog=cell_analog,
                                    finat_element_analog=finat_element_analog,
                                    mesh_analog=mesh_analog,
                                    near_bdy=near_bdy)

            self._fspace_analogs.append(fspace_analog)

        kwargs = dict(self._kwargs)
        conv = FiredrakeMeshmodeConverter(self._cl_ctx,
                                          fspace_analog,
                                          **kwargs)
        self._converters.append(conv)

        return conv


def fd_bind(converter_manager, op, source=None, target=None,
            source_only_near_bdy=False,
            with_refinement=False):
    """
        :arg converter_manager: A :class:`ConverterManager`
        :arg op: The operation
        :arg sources: either
            - A FunctionSpace, which will be the source
            - A pair (FunctionSpace, bdy_id) which will be the source
              (where bdy_id is the boundary which will be the source,
               *None* for the whole mesh)

        :arg targets: either
            - A FunctionSpace, which will be the target
            - A pair (FunctionSpace, bdy_id) which will be the target
              (where bdy_id is the boundary which will be the target,
               *None* for the whole mesh)

        :arg source_only_near_bdy: If *True*, allow conversion of
                                   source function space only
                                   near the give source bdy id, if
                                   one is given.

        :arg with_refinement: If *True*, use refined qbx for source, this
                              is highly recommended
    """

    # Source and target will now be (fspace, bdy_id or *None*)
    if isinstance(source, WithGeometry):
        source = (source, None)
    if isinstance(target, WithGeometry):
        target = (target, None)

    source_converter = \
        converter_manager.get_converter(source[0], bdy_id=source[1],
                                        only_near_bdy=source_only_near_bdy)

    source_connection = SourceConnection(source_converter,
                                         bdy_id=source[1],
                                         with_refinement=with_refinement)

    target_connection = TargetConnection(target[0])
    if target[1] is None:
        target_converter = \
            converter_manager.get_converter(target[0], bdy_id=target[1])
        target_connection.set_converter(target_converter)
        target_connection.set_function_space_as_target()
    else:
        target_connection.set_boundary_as_target(target[1], 'geometric')

    return OpConnection(source_connection, target_connection, op)
