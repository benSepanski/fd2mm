"""Used to raise *UserWarning*s"""
from warnings import warn
import pyopencl as cl
import firedrake as fd
import numpy as np

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential.qbx import QBXLayerPotentialSource


class FiredrakeMeshmodeConverter:
    """
        Conversion :mod:`firedrake` to :mod:`meshmode`

        :arg cl_ctx: An opencl context
        :arg fspace_analog: A :mod:`firedrake_to_pytential.analogs`
                            :class:`FunctionSpaceAnalog`
        :arg bdy_id: If *None*, conversion of functions from
                     firedrake to meshmode is as normal. If a boundary
                     marker is given, however, it converts the function
                     to a meshmode discretization on the given boundary.
                     For the whole boundary, use :mod:`meshmode.mesh`'s
                     :class:`BTAG_ALL`.
    """
    # TODO: Explain kwargs better
    # TODO: Stop using BTAG_ALL, or make synonymous with 'everywhere'
    def __init__(self, cl_ctx, fspace_analog, **kwargs):
        degree = fspace_analog.degree

        # Save this for use in preparing conversion to boundaries
        self._factory = InterpolatoryQuadratureSimplexGroupFactory(degree)

        self._pre_density_discr = Discretization(cl_ctx,
                                                 fspace_analog.meshmode_mesh(),
                                                 self._factory)

        # This is the base qbx source for the whole mesh
        self._domain_qbx = QBXLayerPotentialSource(self._pre_density_discr, **kwargs)

        # Maps a boundary id to a face restriction connection.
        self._bdy_id_to_restriction_connection = {}

        # Maps a boundary id to a refinement connection (already
        # composed with restriction to boundary if boundary id is not *None*)
        self._bdy_id_to_refined_connection = {}

        # Maps a boundary id to the qbx
        self._bdy_id_to_qbx = {}

        # Maps a boundary id to a refined qbx
        self._bdy_id_to_refined_qbx = {}

        # Store fspace analog
        self._fspace_analog = fspace_analog
        self._kwargs = kwargs

    def _prepare_connections(self, bdy_id, with_refinement=False):
        # {{{ Prepare a qbx and restriction connection before any refinement
        if bdy_id not in self._bdy_id_to_qbx:

            # Case 1: bdy_id is *None*, so using the whole mesh
            if bdy_id is None:
                restriction_connection = None

                if self._fspace_analog.near_bdy() is None:
                    qbx = self._domain_qbx
                else:
                    # if fspace is only converted near some boundaries, we can't
                    # do the whole mesh
                    qbx = None

            else:
                from meshmode.discretization.connection import \
                    make_face_restriction

                # Right now we only do exterior boundaries, let's make sure
                # we have exterior faces!
                if self._fspace_analog.exterior_facets.facet_cell.size == 0:
                    if self._fspace_analog.topological_dimension() < \
                            self._fspace_analog.geometric_dimension():
                        warn(" If your mesh is a manifold "
                             " (e.g. a 2-surface in 3-space) "
                             " it probably doesn't have exterior facets at all."
                             " Are you sure you're wanting to convert firedrake "
                             " functions to a boundary? If you're sure, then "
                             " what you're trying to do is currently unsupported.")
                    raise ValueError("No exterior facets listed in"
                                     " <mesh>.exterior_facets.facet_cell."
                                     " In particular, NO BOUNDARY"
                                     " INFORMATION was tagged, so you can't "
                                     " convert onto a boundary.")

                restriction_connection = \
                    make_face_restriction(self._domain_qbx.density_discr,
                                          self._factory, bdy_id)

                qbx = QBXLayerPotentialSource(restriction_connection.to_discr,
                                              **self._kwargs)

            # store in appropriate dicts
            self._bdy_id_to_qbx[bdy_id] = qbx
            self._bdy_id_to_restriction_connection[bdy_id] = restriction_connection

        # }}}

        # {{{ Now prepare a refinement connection, if one does not already exist
        #     and has been requested
        if with_refinement and bdy_id not in self._bdy_id_to_refined_connection:
            # quit if trying to refine whole mesh but only have it near a boundary
            if self._bdy_id_to_qbx[bdy_id] is None:
                self._bdy_id_to_refined_connection[bdy_id] = None
                return

            # Otherwise, refine!
            refined_qbx, refinement_connection = \
                self._bdy_id_to_qbx[bdy_id].with_refinement()

            # Chain together connections if also have a restriction connection
            if bdy_id is not None:
                from meshmode.discretization.connection import \
                    ChainedDiscretizationConnection

                restriction_connection = \
                    self._bdy_id_to_restriction_connection[bdy_id]
                refinement_connection = ChainedDiscretizationConnection([
                    restriction_connection,
                    refinement_connection])

            self._bdy_id_to_refined_qbx[bdy_id] = refined_qbx
            self._bdy_id_to_refined_connection[bdy_id] = refinement_connection
        # }}}

    def flatten_refinement_chain(self, queue, bdy_id):
        """
            Flattens the chain restriction->refinement into a single
            connection for the given boundary id :arg:`bdy_id`
        """
        self._prepare_connections(bdy_id, with_refinement=True)
        if bdy_id is None:
            # Nothing to do because no face restriction
            return

        from meshmode.discretization.connection import flatten_chained_connection

        # Flatten chain and store in dict
        refinement_connection = self._bdy_id_to_refined_connection[bdy_id]
        refined_connection = flatten_chained_connection(queue, refinement_connection)
        self._bdy_id_to_refined_connection[bdy_id] = refined_connection

    def get_qbx(self, bdy_id=None, with_refinement=False):
        """
            Returns a :class:`QBXLayerPotentialSource` for the given
            :arg:`bdy_id`, refined if :arg:`with_refinement` is *True*
        """
        self._prepare_connections(bdy_id, with_refinement=with_refinement)

        if with_refinement:
            return self._bdy_id_to_refined_qbx[bdy_id]
        return self._bdy_id_to_qbx[bdy_id]

    def get_connection(self, bdy_id=None, with_refinement=None):
        """
            Returns a connection (as in :mod:`meshmode.discretization.connection`)
            from the function space analog of this converter (:attr:`_fspace_analog`)
            to a discretization on the given :arg:`bdy_id`, refined if
            :arg:`with_refinement` is *True*
        """
        self._prepare_connections(bdy_id, with_refinement=with_refinement)

        if with_refinement:
            return self._bdy_id_to_refined_connection[bdy_id]
        else:
            if bdy_id is None:
                return None
            return self._bdy_id_to_restriction_connection[bdy_id]

    def can_convert(self, function_space, bdy_id=None):
        """
            :arg function_space: A :mod:`firedrake` FunctionSpace of Function
            :arg bdy_id: A boundary marker for the mesh :arg:`function_or_space`
                         lives on, or *None* if converting onto the whole
                         mesh.

            Returns *True* if and only if this object can convert functions
            on the given function space to a pytential mesh on the
            either the whole mesh, or the given :arg:`bdy_id`
        """
        return self._fspace_analog.is_analog(function_space, near_bdy=bdy_id)

    def convert(self, queue, weights, firedrake_to_meshmode=True,
                bdy_id=None, with_refinement=None):
        """
            Returns converted weights as an *np.array*

            Firedrake->meshmode conversion, converts to source mesh
            meshmode->Firedrake requires domain mesh == source mesh

            :arg queue: The pyopencl queue
                        NOTE: May pass *None* unless source is an interpolation
                              onto the boundary of the domain mesh
                        NOTE: Must be created from same cl_ctx this object
                              created with
            :arg weights:
                - An *np.array* with the weights representing the function or
                  discretization
                - a Firedrake :class:`Function`
        """
        if queue is None and bdy_id is not None:
            raise ValueError("""Cannot pass *None* for :arg:`queue` if the
                              source mesh is not the whole domain""")
        if not firedrake_to_meshmode and bdy_id is not None:
            raise ValueError("""Cannot convert from meshmode boundary to
                             firedrake""")

        # {{{ Convert data to np.array if necessary
        data = weights
        if isinstance(weights, fd.Function):
            assert firedrake_to_meshmode
            if not self._fspace_analog.is_analog(weights.function_space(),
                                                 near_bdy=bdy_id):
                raise ValueError("Function not on valid function space and bdy id"
                                 " for this class's FunctionSpaceAnalog")
            data = weights.dat.data
        if isinstance(weights, cl.array.Array):
            assert not firedrake_to_meshmode
            if queue is None:
                raise ValueError("Must supply queue for meshmode to firedrake"
                                 " conversion")
            data = weights.get(queue=queue)
        elif not isinstance(data, np.ndarray):
            raise ValueError("weights type not one of"
                             " Firedrake.Function, np.array, cl.array]")
        # }}}

        # Get the array with the re-ordering applied
        data = self._fspace_analog.reorder_nodes(
            data, firedrake_to_meshmode=firedrake_to_meshmode)

        # {{{ if interpolation onto the source / refinement is required,
        #     do so

        connection = self.get_connection(bdy_id, with_refinement)

        if connection is not None:
            # if a vector function space, data is a np.array of arrays
            if len(data.shape) > 1:
                data_array = []
                for arr in data:
                    new_arr = connection(queue, cl.array.to_device(queue, arr))
                    data_array.append(new_arr.get(queue=queue))
                data = np.array(data_array)
            # Else is just an array
            else:
                data = cl.array.to_device(queue, data)
                data = connection(queue, data).with_queue(queue)
                data = data.get(queue=queue)
        # }}}

        return data

    def converting_entire_mesh(self):
        """
            Returns *True* iff converting the entire mesh, i.e.
            not only converting near some portion of the boundary
        """
        self._prepare_connections(None, with_refinement=False)
        return self._bdy_id_to_qbx[None] is not None
