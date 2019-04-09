import numpy as np
import pyopencl as cl
from firedrake import SpatialCoordinate, Function, \
    VectorFunctionSpace, FunctionSpace, project
from firedrake_to_pytential import FiredrakeMeshmodeConnection
from pytential import bind
from pytential.target import PointsTarget
from warnings import warn


class OpConnection:
    """
        The idea is to make an easier interface for defining
        operations quickly.
        Rather than have the user manage all of the convergence,
        simply evaluate an operation on a firedrake mesh or
        boundary points
    """

    def __init__(self, cl_ctx, function_space, op=None,
                 targets=None, ambient_dim=None, source_bdy_id=None, **kwargs):
        """
            :arg op: the operation to be evaluated.
            NOTE: All operations are bound using a QBXLayerPotentialSource

            :arg targets:
             - an *int*, the target will be the
               boundary ids at :arg:`targets`.
             - If an iterable of *int* types, then
               the target will be any boundary which
               has one of the given ids

             WARNING: Currently, only exterior facet ids are
                      supported

             WARNING: In either case, the pytential
                      op is just passed a collection of points
                      for evaluation. In particular, any attempt
                      at normal derivative evaluation along
                      the target mesh will fail.

            :arg function_space: Function space used for evaluation.
                If not a 'DG' space, then converted to a 'DG' space
                for evaluation.

            For other args, see :class:`FiredrakeMeshmodeConnection`
        """
        # Handle function space
        m = function_space.mesh()

        self.space_is_dg = True
        self.dg_function_space = function_space
        if function_space.ufl_element() != 'Discontinuous Lagrange':
            warn("Creating DG function space for evaluation")
            self.space_is_dg = False
            degree = function_space.ufl_element().degree()

            if not function_space.shape:
                self.dg_function_space = FunctionSpace(m, "DG", degree=degree)
            elif len(function_space.shape) == 1:
                self.dg_function_space = VectorFunctionSpace(
                    m, "DG", degree=degree, dim=function_space.shape[0])
            elif len(function_space.shape) > 1:
                raise ValueError("TensorFunctionSpace not yet supported")
                

        # Connection to meshmode version
        self.to_meshmode = FiredrakeMeshmodeConnection(
            cl_ctx, self.dg_function_space, ambient_dim=ambient_dim,
            source_bdy_id=source_bdy_id, **kwargs)

        # {{{ Handle targets

        # if just passed an int, convert to an iterable of ints
        # so that just one case to deal with
        if isinstance(targets, int):
            targets = [targets]
        target_markers = set(targets)

        # Check that boundary ids are valid
        if not target_markers <= set(m.exterior_facets.unique_markers):
            warn("The following boundary ids are not exterior facet ids: %s" %
                 (target_markers - set(m.exterior_facets.unique_markers)))

        if not target_markers & set(m.exterior_facets.unique_markers):
            raise ValueError("No boundary ids are exterior facet ids")

        self.target_indices = set()
        for marker in target_markers:
            self.target_indices |= set(
                function_space.boundary_nodes(marker, 'geometric'))
        self.target_indices = np.array(list(self.target_indices), dtype=np.int32)

        # Get coordinates of nodes
        if ambient_dim is None:
            ambient_dim = m.geometric_dimension()
        xx = SpatialCoordinate(m)
        function_space_dim = VectorFunctionSpace(
            m,
            function_space.ufl_element().family(),
            degree=function_space.ufl_element().degree(),
            dim=ambient_dim)
        coords = Function(function_space_dim).interpolate(xx)
        coords = np.real(coords.dat.data)
        self.target_pts = coords[self.target_indices]
        # change from [nnodes][ambient_dim] to [ambient_dim][nnodes]
        self.target_pts = np.transpose(self.target_pts).copy()

        # }}}

        self.bound_op = None
        if op:
            self.set_op(op)

    def set_op(self, op):
        qbx = self.to_meshmode.qbx_map['source']
        self.bound_op = bind((qbx, PointsTarget(self.target_pts)), op)


    def __call__(self, queue, result_function=None,
                 out_function_space=None, **kwargs):
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
        if out_function_space is None:
            out_function_space = self.to_meshmode.fd_function_space

        new_kwargs = {}
        for key in kwargs:
            if isinstance(kwargs[key], Function):
                # Make sure in dg space
                if self.space_is_dg:
                    fntn = kwargs[key]
                else:
                    fntn = project(kwargs[key], self.dg_function_space,
                                   use_slate_for_inverse=False)

                # Convert function to array with pytential ordering
                pyt_fntn = self.to_meshmode(queue, fntn)

                # Put on queue
                new_kwargs[key] = cl.array.to_device(queue, pyt_fntn)
            else:
                new_kwargs[key] = kwargs[key]

        # Perform operation and take result off queue
        result = self.bound_op(queue, **new_kwargs)
        result = result.get(queue=queue)

        # Create firedrake function
        if result_function is None:
            result_function = Function(out_function_space)
            result_function.dat.data[:] = 0.0
        result_function.dat.data[self.target_indices] = result

        return result_function
