from warnings import warn
import pyopencl as cl
import firedrake as fd
import numpy as np

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential.qbx import QBXLayerPotentialSource


thresh = 1e-8


class FiredrakeMeshmodeConverter:
    """
        Conversion :mod:`firedrake` to :mod:`meshmode`
    """
    def __init__(self, cl_ctx, dg_fspace_analog, bdy_id=None, **kwargs):

        degree = dg_fspace_analog.degree()
        fine_order = kwargs.get('fine_order', degree)
        fmm_order = kwargs.get('fmm_order', degree)
        qbx_order = kwargs.get('qbx_order', degree)
        with_refinement = kwargs.get('with_refinement', False)

        factory = InterpolatoryQuadratureSimplexGroupFactory(degree)

        pre_density_discr = Discretization(
            cl_ctx,
            dg_fspace_analog.meshmode_mesh(),
            factory)

        self._domain_qbx = QBXLayerPotentialSource(pre_density_discr,
                                                   fine_order=fine_order,
                                                   qbx_order=qbx_order,
                                                   fmm_order=fmm_order)

        if bdy_id is not None:
            from meshmode.discretization.connection import \
                make_face_restriction

            self._domain_to_source = make_face_restriction(
                self._domain_qbx.density_discr,
                factory, bdy_id)

            self._source_qbx = QBXLayerPotentialSource(
                self._domain_to_source.to_discr,
                fine_order=fine_order,
                qbx_order=qbx_order,
                fmm_order=fmm_order)
        else:
            self._domain_to_source = None
            self._source_qbx = self._domain_qbx

        self._dg_fspace_analog = dg_fspace_analog
        self._bdy_id = bdy_id

        if with_refinement:
            if self._domain_to_source is None:
                warn("Only refine when mesh has codim 1")
            else:
                self._source_qbx, _ = self._source_qbx.with_refinement()

    def can_convert(self, function_space, bdy_id=None):
        return (self._dg_fspace_analog.is_analog(function_space)
                and self._bdy_id == bdy_id)

    def convert(self, queue, weights, firedrake_to_meshmode=True):
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
        if queue is None and self._domain_to_source is not None:
            raise ValueError("""Cannot pass *None* for :arg:`queue` if the
                              source mesh is not the whole domain""")
        if not firedrake_to_meshmode:
            if self._domain_to_source is not None:
                raise ValueError("""Cannot convert from meshmode boundary to
                                 firedrake""")

        # {{{ Convert data to np.array if necessary
        data = weights
        if isinstance(weights, fd.Function):
            assert firedrake_to_meshmode
            if not self._dg_fspace_analog.is_valid(weights.function_space()):
                raise ValueError("Function not on valid function space for"
                                 " given this class's DGFunctionSpaceAnalog")
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
        data = self._dg_fspace_analog.reorder_nodes(
            data, firedrake_to_meshmode=firedrake_to_meshmode)

        # {{{ if interpolation onto the source is required, do so
        if self._domain_to_source is not None:
            # if a vector function space, data is a np.array of arrays
            if len(data.shape) > 1:
                data_array = []
                for arr in data:
                    data_array.append(
                        self._domain_to_source(
                            queue, cl.array.to_device(queue, arr)).get(queue=queue)
                    )
                data = np.array(data_array)
            else:
                data = cl.array.to_device(queue, data)
                data = \
                    self._domain_to_source(queue, data).\
                    with_queue(queue).get(queue=queue)
        # }}}

        return data
