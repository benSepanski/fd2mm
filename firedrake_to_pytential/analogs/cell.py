import numpy as np
import numpy.linalg as la

from FIAT.reference_element import Simplex
from modepy import tools

from firedrake_to_pytential.analogs import Analog


def _get_affine_mapping(reference_vects, vects):
    r"""
    Returns (mat, shift),
    a matrix *mat* and vector *shift* which maps the
    *i*th vector in :arg:`v` to the *i*th vector in :arg:`w` by

    ..math::

        A vi + b -> wi, \qquad A = mat, b = shift

    :arg reference_vects: An np.array of *n* vectors of dimension *ref_dim*
    :arg vects: An np.array of *n* vectors of dimension *dim*, with
                *ref_dim* <= *dim*.

        NOTE : Should be shape (ref_dim, nvectors), (dim, nvectors) respectively.

    *mat* will have shape (dim, ref_dim), *shift* will have shape (dim)
    """
    # Make sure both have same number of vectors
    ref_dim, num_vects = reference_vects.shape
    assert num_vects == vects.shape[1]

    # Make sure d1 <= d2 (see docstring)
    dim = vects.shape[0]
    assert ref_dim <= dim

    # If there is only one vector, set M = I, b = vect - reference
    if num_vects == 1:
        mat = np.eye(dim, ref_dim)
        shift = vects[:, 0] - np.matmul(mat, reference_vects[:, 0])
    else:
        ref_span_vects = reference_vects[:, 1:] - reference_vects[:, 0, np.newaxis]
        span_vects = vects[:, 1:] - vects[:, 0, np.newaxis]
        mat = la.solve(ref_span_vects, span_vects)
        shift = -np.matmul(mat, reference_vects[:, 0]) + vects[:, 0]

    return mat, shift


class SimplexCellAnalog(Analog):
    """
        cell analog, for simplices only
    """
    def __init__(self, cell):
        """
            :arg cell: a :mod:`FIAT` :class:`reference_element`
        """
        # Ensure this cell is actually a simplex
        assert isinstance(cell, Simplex)

        super(SimplexCellAnalog, self).__init__(cell)

        reference_vertices = np.array(cell.vertices)

        dim = reference_vertices.shape[0] - 1
        # Stored as (nunit_vertices, dim)
        self._unit_vertices = tools.unit_vertices(dim)

        # Maps firedrake reference nodes to :mod:`meshmode`
        # unit nodes
        self._mat, self._shift = _get_affine_mapping(reference_vertices.T,
                                                     self._unit_vertices.T)

    def make_points(self, dim, entity_id, order):
        """
            as called by a cell in firedrake, but converted
            to :mod:`modepy` unit nodes
        """
        points = self.analog().make_points(dim, entity_id, order)
        if not points:
            return points
        points = np.array(points)
        # Points is (nvertices, dim) so have to transpose
        return (np.matmul(self._mat, points.T) + self._shift[:, np.newaxis]).T

    def unit_vertices(self):
        """
            Returns unit vertices (that is, in :mod:`meshmode` coordinates)
            as an (num_unit_vertices, dim) shaped *np.ndarray*
        """
        return self._unit_vertices
