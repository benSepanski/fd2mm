import numpy as np
import numpy.linalg as la
import six

from finat.fiat_elements import DiscontinuousLagrange, Lagrange
from numpy.polynomial.polynomial import polyvander, polyval, polyval2d, polyval3d

from fd2mm.analog import Analog
from fd2mm.cell import SimplexCellAnalog


__doc__ = """
.. autoclass:: FinatElementAnalog
    :members:
"""


class FinatElementAnalog(Analog):
    """
    An analog for a FInAT element, usually called on
    ``some_function_space.finat_element``
    """
    def __init__(self, finat_element):
        """
            :arg finat_element: A FInAT element
            :raises TypeError: If FInAT element is not of type
                               :class:`finat.fiat_elements.Lagrange` or
                               :class:`finat.fiat_elements.DiscontinuousLagrange`
        """
        # {{{ Parse input

        # Check types
        if not isinstance(finat_element, Lagrange) \
                and not isinstance(finat_element, DiscontinuousLagrange):
            raise TypeError(":arg:`finat_element` must be of type"
                            " finat.fiat_elements.Lagrange"
                            " finat.fiat_elements.DiscontinuousLagrange")

        assert finat_element.mapping == 'affine', \
            "FInAT element must use affine mappings of the bases"
        # }}}

        super(FinatElementAnalog, self).__init__(finat_element)

        self.cell_a = SimplexCellAnalog(finat_element.cell)

        self._unit_nodes = None
        self._unit_vertex_indices = None
        self._flip_matrix = None

        # See :meth:`map_points`
        self._vandermonde_inv = None

    def _compute_unit_vertex_indices_and_nodes(self):
        """
            Explicitly compute the unit nodes, as well as the
            unit vertex indices
        """
        if self._unit_nodes is None or self._unit_vertex_indices is None:
            # {{{ Compute unit nodes
            node_nr_to_coords = {}
            unit_vertex_indices = []

            # Get unit nodes
            for dim, element_nrs in six.iteritems(
                    self.analog().entity_support_dofs()):
                for element_nr, node_list in six.iteritems(element_nrs):
                    # Get the nodes on the element (in meshmode reference coords)
                    pts_on_element = self.cell_a.make_points(
                        dim, element_nr, self.analog().degree)
                    # Record any new nodes
                    i = 0
                    for node_nr in node_list:
                        if node_nr not in node_nr_to_coords:
                            node_nr_to_coords[node_nr] = pts_on_element[i]
                            i += 1
                            # If is a vertex, store the index
                            if dim == 0:
                                unit_vertex_indices.append(node_nr)

            # store vertex indices
            self._unit_vertex_indices = np.array(sorted(unit_vertex_indices))

            # Convert unit_nodes to array, then change to (dim, nunit_nodes)
            # from (nunit_nodes, dim)
            unit_nodes = np.array([node_nr_to_coords[i] for i in
                                   range(len(node_nr_to_coords))])
            self._unit_nodes = unit_nodes.T.copy()

            # }}}

    def dim(self):
        """
            :return: The dimension of the FInAT element's cell
        """
        return self.cell_a.analog().get_dimension()

    def unit_vertex_indices(self):
        """
            :return: An array of shape *(dim+1,)* of indices
                     so that *self.unit_nodes()[self.unit_vertex_indices()]*
                     are the vertices of the reference element.
        """
        self._compute_unit_vertex_indices_and_nodes()
        return self._unit_vertex_indices

    def unit_nodes(self):
        """
            :return: The unit nodes used by the FInAT element mapped
                     onto the appropriate :mod:`modepy` `reference
                     element <https://documen.tician.de/modepy/nodes.html>`_
                     as an array of shape *(dim, nunit_nodes)*.
        """
        self._compute_unit_vertex_indices_and_nodes()
        return self._unit_nodes

    def nunit_nodes(self):
        """
            :return: The number of unit nodes.
        """
        return self.unit_nodes().shape[1]

    def flip_matrix(self):
        """
            :return: The matrix which should be applied to the
                     *(dim, nunitnodes)*-shaped array of nodes corresponding to
                     an element in order to change orientation - <-> +.

                     The matrix will be *(dim, dim)* and orthogonal with
                     *np.float64* type entries.
        """
        if self._flip_matrix is None:
            # This is very similar to :mod:`meshmode` in processing.py
            # the function :function:`from_simplex_element_group`, but
            # we needed to use firedrake nodes

            from modepy.tools import barycentric_to_unit, unit_to_barycentric

            # Generate a resampling matrix that corresponds to the
            # first two barycentric coordinates being swapped.

            bary_unit_nodes = unit_to_barycentric(self.unit_nodes())

            flipped_bary_unit_nodes = bary_unit_nodes.copy()
            flipped_bary_unit_nodes[0, :] = bary_unit_nodes[1, :]
            flipped_bary_unit_nodes[1, :] = bary_unit_nodes[0, :]
            flipped_unit_nodes = barycentric_to_unit(flipped_bary_unit_nodes)

            from modepy import resampling_matrix, simplex_best_available_basis

            flip_matrix = resampling_matrix(
                simplex_best_available_basis(self.dim(), self.analog().degree),
                flipped_unit_nodes, self.unit_nodes())

            flip_matrix[np.abs(flip_matrix) < 1e-15] = 0

            # Flipping twice should be the identity
            assert la.norm(
                np.dot(flip_matrix, flip_matrix)
                - np.eye(len(flip_matrix))) < 1e-13

            self._flip_matrix = flip_matrix

        return self._flip_matrix

    def make_resampling_matrix(self, element_grp):
        """
            :arg element_grp: A
                :class:`meshmode.discretization.InterpolatoryElementGroupBase` whose
                basis functions span the same space as the FInAT element.
            :return: A matrix which resamples a function sampled at
                     the firedrake unit nodes to a function sampled at
                     *element_grp.unit_nodes()* (by matrix multiplication)
        """
        from meshmode.discretization import InterpolatoryElementGroupBase
        assert isinstance(element_grp, InterpolatoryElementGroupBase), \
            "element group must be an interpolatory element group so that" \
            " can redistribute onto its nodes"

        from modepy import resampling_matrix
        return resampling_matrix(element_grp.basis(),
                                 new_nodes=element_grp.unit_nodes,
                                 old_nodes=self.unit_nodes())
