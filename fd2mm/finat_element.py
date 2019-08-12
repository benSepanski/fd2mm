import numpy as np
import numpy.linalg as la
import six

from finat.fiat_elements import DiscontinuousLagrange, Lagrange
from numpy.polynomial.polynomial import polyvander, polyval, polyval2d, polyval3d

from fd2mm.analog import Analog
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
            Returns the dimension of the cell
        """
        return self.cell_a.analog().get_dimension()

    def unit_vertex_indices(self):
        self._compute_unit_vertex_indices_and_nodes()
        return self._unit_vertex_indices

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
        """
            :arg points: a (:meth:`dim`, npoints) array of points
        """
        degree = self.analog().degree
        npoints = points.shape[1]

        # In 1-D, this is easy
        if self.dim() == 1:
            return polyvander(points[0], degree)

        # {{{ Traverse over all multi-indices with sum less than *degree*

        # Number of unit nodes should match number of basis elts in polynomial,
        # (otherwise we have the wrong amount of information)
        num_multi_indices = self.unit_nodes().shape[1]

        vandermonde = np.zeros((npoints, num_multi_indices))
        # 1-D vandermonde mats, i.e. for each point p: 1, p, p^2, p^3, p^4, ...
        # with shape (npoints, dim, degree+1)
        one_dim_vanders = polyvander(points.T, degree)

        multi_ndx = [0] * self.dim()
        ibasis_elt = 0  # index of basis element
        while ibasis_elt < num_multi_indices:
            for pt_ndx in range(npoints):
                vandermonde[pt_ndx, ibasis_elt] = \
                    np.prod([vander[power] for vander, power in
                             zip(one_dim_vanders[pt_ndx], multi_ndx)])

            # Get next multi-index
            multi_ndx[-1] += 1
            ibasis_elt += 1

            j = len(multi_ndx) - 1
            j_limit = degree - sum(multi_ndx[:j])
            while j > 0 and multi_ndx[j] > j_limit:
                multi_ndx[j] = 0
                j_limit -= multi_ndx[j-1]
                multi_ndx[j-1] += 1
                j -= 1

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

        if self._vandermonde_inv is None:
            r"""
                For 1 <= i <= :meth:`dim` we have a degree d map T_i mapping the unit
                nodes of this object onto nodes. We wind up with a matrix problem

                .. math::

                    V [coeffs] = nodes^T

                We compute and store V^{-1} so that we can
                compute the coefficients quickly. Note V will be a vandermonde matrix
                based on the unit nodes

                (e.g. if d is 2, then the relevant basis would be
                {1, x, y, xy, x^2, y^2}. Each row of V is associated to a unit node
                *un*, and each column of that row is one of the basis polynomials
                evaluated at *un*)
            """
            vandermonde = self.vandermonde(self.unit_nodes())
            self._vandermonde_inv = np.linalg.inv(vandermonde)

        coeffs = np.matmul(self._vandermonde_inv, nodes.T)

        # apply polynomial to points
        return np.matmul(self.vandermonde(points), coeffs).T

    def make_resampling_matrix(self, element_grp):
        from meshmode.discretization import InterpolatoryElementGroupBase
        assert isinstance(element_grp, InterpolatoryElementGroupBase), \
            "element group must be an interpolatory element group so that" \
            " can redistribute onto its nodes"

        from modepy import resampling_matrix
        return resampling_matrix(element_grp.basis(),
                                 new_nodes=element_grp.unit_nodes,
                                 old_nodes=self.unit_nodes())
