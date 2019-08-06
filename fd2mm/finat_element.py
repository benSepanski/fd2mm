import numpy as np
import numpy.linalg as la
import six

from finat.fiat_elements import DiscontinuousLagrange, Lagrange
from numpy.polynomial.polynomial import polyvander, polyval, polyval2d, polyval3d

from fd2mm import Analog
from fd2mm.cell import SimplexCellAnalog


class FinatElementAnalog(Analog):
    """
        Analog for a :mod:`finat` fiat element.

        e.g. given a :mod:`firedrake` function space *V*, you could
        call
        ```
        finat_element_analog = FinatElementAnalog(V.finat_element)
        ```
    """

    def __init__(self, finat_element):
        """
            :arg:`finat_element` must be either Lagrange or
            DiscontinuousLagrange, else will raise *TypeError*

            :arg finat_element: A :mod:`finat` fiat element
        """
        # {{{ Parse input

        # Check types
        if not isinstance(finat_element, Lagrange) \
                and not isinstance(finat_element, DiscontinuousLagrange):
            raise TypeError(":arg:`finat_element` must be of type"
                            " finat.fiat_elements.Lagrange"
                            " finat.fiat_elements.DiscontinuousLagrange")
        # }}}

        super(FinatElementAnalog, self).__init__(finat_element)

        self.cell_a = SimplexCellAnalog(finat_element.cell)

        self._unit_nodes = None
        self._unit_vertex_indices = None
        self._flip_matrix = None

        # The appropriate dimension polynomial evaluation method from numpy
        self._poly_val_method = None

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
            self._unit_vertex_indices = np.array(unit_vertex_indices)

            # Convert unit_nodes to array, then change to (dim, nunit_nodes)
            # from (nunit_nodes, dim)
            unit_nodes = np.array([node_nr_to_coords[i] for i in
                                   range(len(node_nr_to_coords))])
            self._unit_nodes = unit_nodes.T.copy()

            # }}}

    def dim(self):
        """
            Returns the dimension of the cell
        """
        return self.cell_a.analog().get_dimension()

    def unit_nodes(self):
        """
            gets unit nodes (following :mod:`modepy` rules for the reference simplex)
            as (dim, nunit_nodes) shape
        """
        self._compute_unit_vertex_indices_and_nodes()
        return self._unit_nodes

    def nunit_nodes(self):
        return self.unit_nodes().shape[1]

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

    def vandermonde(self, points):
        degree = self.analog().degree

        # In 1-D, this is easy
        if self.dim() == 1:
            return polyvander(points[0], degree)

        # {{{ Traverse over all multi-indices with sum less than *degree*

        # Number of unit nodes should match number of basis elts in polynomial,
        # otherwise we have the wrong amount of information
        num_multi_indices = self.unit_nodes().shape[1]

        # Vandermonde will be (npoints, num_multi_indices)
        vandermonde = np.zeros((points.shape[1], num_multi_indices))
        # 1-D vandermonde mats, i.e. for each point p: 1, p, p^2, p^3, p^4, ...
        # with shape (npoints, :meth:`dim`, degree)
        one_dim_vanders = polyvander(points.T, degree)

        multi_ndx = [0] * self.dim()
        i = 0
        while i < num_multi_indices:
            for pt_ndx in range(points.shape[1]):
                vandermonde[pt_ndx, i] = \
                    np.prod([vander[power] for vander, power in
                             zip(one_dim_vanders[pt_ndx], multi_ndx)])

            # Get next multi-index
            multi_ndx[-1] += 1
            i += 1

            j = len(multi_ndx) - 1
            j_limit = degree - sum(multi_ndx[:j])
            while multi_ndx[j] >= j_limit:
                multi_ndx[j] = 0
                multi_ndx[j-1] += 1
                j -= 1
                j_limit += multi_ndx[j]
        # }}}

        return vandermonde

    def map_points(self, nodes, points):
        """
            :arg nodes: a (:meth:`dim`, nunitnodes) numpy array describing where the
                        unit nodes map to
            :arg points: a (:meth:`dim`, npoints) numpy array of points on the
                         reference cell

            Returns the locations of arg:`points` according to the map described
            by :arg:`nodes`, i.e there is a :attr:`self.analog().degree`-degree
            mapping :math:`T` from :meth:`unit_nodes` onto :arg:`nodes`,
            and this function computes :math:`T(`:arg:`points`:math:`)`
        """
        # For 1 <= i <= :meth:`dim` we have a degree d map T_i mapping the unit
        # nodes of this object onto nodes. We wind up with a matrix problem
        # A [coeffs] = nodes^T. We compute and store A^{-1} so that we can
        # compute the coefficients quickly. Note A will be a vandermonde matrix

        degree = self.analog().degree
        # Compute vandermonde matrix inverse
        if self._vandermonde_inv is None:
            degree_arr = [degree for _ in range(self.dim())]

            # Get appropriate method from numpy
            if self.dim() == 1:
                self._poly_val_method = polyval
                degree_arr, = degree_arr  # unpack if in 1d
            elif self.dim() == 2:
                self._poly_val_method = polyval2d
            elif self.dim() == 3:
                self._poly_val_method = polyval3d
            else:
                raise ValueError("Geometric dimension must be 1, 2, or 3")

            self._vandermonde_inv = np.linalg.inv(self.vandermonde(nodes))

        coeffs = np.matmul(self._vandermonde_inv, nodes.T)

        # apply polynomial to points
        return np.matmul(self.vandermonde(points), coeffs).T
