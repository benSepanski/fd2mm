import numpy as np
import numpy.linalg as la
import six

from finat.fiat_elements import DiscontinuousLagrange, Lagrange

from firedrake_to_pytential.analogs import Analog
from firedrake_to_pytential.analogs.cell import SimplexCellAnalog


class FinatElementAnalog(Analog):
    """
        Analog for a :mod:`finat` fiat element.

        e.g. given a :mod:`firedrake` function space *V*, you could
        call
        ```
        finat_element_analog = FinatElementAnalog(V.finat_element)
        ```
    """

    def __init__(self, finat_element, cell_analog=None):
        """
            :arg:`finat_element` must be either Lagrange or
            DiscontinuousLagrange, else will raise *TypeError*

            :arg finat_element: A :mod:`finat` fiat element
            :arg cell_analog: Either a :class:`SimplexCellAnalog` associated to the
                              :attr:`cell` of :arg:`finat_element`, or *None*, in
                              which case a :class:`SimplexCellAnalog` is constructed.
        """
        if not isinstance(finat_element, Lagrange) \
                and not isinstance(finat_element, DiscontinuousLagrange):
            raise TypeError("Finat element must of type"
                            " finat.fiat_elements.Lagrange"
                            " finat.fiat_elements.DiscontinuousLagrange")

        self._unit_nodes = None
        self._barycentric_unit_nodes = None
        self._flip_matrix = None

        if finat_element.mapping != 'affine':
            raise ValueError("Non-affine mappings are currently unsupported")

        if cell_analog is None:
            # Construct cell analog if needed
            cell_analog = SimplexCellAnalog(finat_element.cell)
        else:
            # Else make sure the cell analog is an analog the right cell!
            assert cell_analog.analog() == finat_element.cell

        self._cell_analog = cell_analog
        super(FinatElementAnalog, self).__init__(finat_element)

    def unit_nodes(self):
        """
            gets unit nodes (following :mod:`modepy` rules for the reference simplex)
            as (dim, nunit_nodes)
        """
        if self._unit_nodes is None:
            node_nr_to_coords = {}

            # {{{ Get unit nodes (in :mod:`meshmode` coordinates)
            for dim, element_nrs in six.iteritems(
                    self.analog().entity_support_dofs()):
                for element_nr, node_list in six.iteritems(element_nrs):
                    pts_on_element = self._cell_analog.make_points(
                        dim, element_nr, self.analog().degree)
                    i = 0
                    for node_nr in node_list:
                        if node_nr not in node_nr_to_coords:
                            node_nr_to_coords[node_nr] = pts_on_element[i]
                            i += 1
            # }}}

            # Convert unit_nodes to array, then change to (dim, nunit_nodes)
            # from (nunit_nodes, dim)
            unit_nodes = np.array([node_nr_to_coords[i] for i in
                                   range(len(node_nr_to_coords))])
            self._unit_nodes = unit_nodes.T.copy()

        return self._unit_nodes

    def barycentric_unit_nodes(self):
        """
            Gets unit nodes in barycentric coordinates
            as (dim, nunit_nodes)
        """
        if self._barycentric_unit_nodes is None:
            # unit vertices is (nunit_vertices, dim), change to
            # (dim, nunit_vertices)
            unit_vertices = self._cell_analog.unit_vertices().T.copy()
            r"""
                If the vertices of the unit simplex are

                ..math::

                v_0,\dots, v_n

                we want to write some vector x as

                ..math::

                \sum b_i v_i, \qquad \sum b_i = 1

                In particular, we have

                ..math::

                b_0 = 1 - \sum b_i
                \implies
                x = \sum_{i\geq 1} (b_iv_i) + b_0v_0
                  = \sum_{i\geq 1} (b_iv_i) + (1 - \sum_{i\geq 1} b_i)v_0
                \implies
                x - v_0  = \sum_{i\geq 1} b_i(v_i - v_0)
            """
            # The columns of the matrix are
            # [v_1 - v_0, v_2 - v_0, \dots, v_d - v_0]
            unit_vert_span_vects = \
                unit_vertices[:, 1:] - unit_vertices[:, 0, np.newaxis]

            # Solve the matrix. This will map a vector \sum b_i(v_i - v_0)
            # to the vector (b_1, b_2, \dots, b_d)^T
            to_span_vects_coordinates = la.inv(unit_vert_span_vects)

            # [node_1 - v_0, node_2 - v_0, \dots, node_n - v_0]
            shifted_unit_nodes = self.unit_nodes() - unit_vertices[:, 0, np.newaxis]

            # shape (dim, unit_nodes), columns look like
            # [ last d bary coords of node 1, ..., last d bary coords of node n]
            bary_nodes = np.matmul(to_span_vects_coordinates, shifted_unit_nodes)

            # barycentric coordinates are of dimension d+1, so make an
            # array in which to store them
            dim, nunit_nodes = self.unit_nodes().shape
            self._barycentric_unit_nodes = np.ones((dim + 1, nunit_nodes))

            # compute b_0 for each unit node
            self._barycentric_unit_nodes[0] -= np.einsum("ij->j", bary_nodes)

            # store b_1, b_2,\dots, b_d for each node
            self._barycentric_unit_nodes[1:] = bary_nodes

        return self._barycentric_unit_nodes

    def flip_matrix(self):
        """
            Returns the matrix which should be applied to the
            (dim, nnodes)-shaped array of nodes corresponding to
            an element in order to change orientation - <-> +.

            The matrix will be (dim, dim) and orthogonal with
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
            dim = self.analog().cell.get_dimension()

            flip_matrix = resampling_matrix(
                simplex_best_available_basis(dim, self.analog().degree),
                flipped_unit_nodes, self.unit_nodes())

            flip_matrix[np.abs(flip_matrix) < 1e-15] = 0

            # Flipping twice should be the identity
            assert la.norm(
                np.dot(flip_matrix, flip_matrix)
                - np.eye(len(flip_matrix))) < 1e-13

            self._flip_matrix = flip_matrix

        return self._flip_matrix
