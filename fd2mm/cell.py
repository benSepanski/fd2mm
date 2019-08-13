import numpy as np

from FIAT.reference_element import Simplex
from modepy import tools

from fd2mm.analog import Analog
from fd2mm.utils import get_affine_mapping


__doc__ = """
.. autoclass:: SimplexCellAnalog
    :members:

"""


class SimplexCellAnalog(Analog):
    """
        Analog of a :mod:`FIAT` simplex cell.
    """
    def __init__(self, cell):
        """
            :arg cell: a :class:`fiat.FIAT.reference_element.Simplex`.
        """
        # Ensure this cell is actually a simplex
        assert isinstance(cell, Simplex)

        super(SimplexCellAnalog, self).__init__(cell)

        # Stored as (dim, nunit_vertices)
        self._unit_vertices = tools.unit_vertices(cell.get_dimension()).T

        # Maps firedrake reference vertices to :mod:`meshmode`
        # unit vertices by x -> Ax + b, where A is :attr:`_mat`
        # and b is :attr:`_shift`
        reference_vertices = np.array(cell.vertices).T
        self._mat, self._shift = get_affine_mapping(reference_vertices,
                                                    self._unit_vertices)

    def make_points(self, dim, entity_id, order):
        """
            Args are exactly as in
            :meth:`fiat.FIAT.reference_element.Cell.make_points`, however
            the unit nodes are (affinely) mapped to :mod:`modepy`
            `unit coordinates <https://documen.tician.de/modepy/nodes.html>`_.
        """
        points = self.analog().make_points(dim, entity_id, order)
        if not points:
            return points
        points = np.array(points)
        # Points is (nvertices, dim) so have to transpose
        return (np.matmul(self._mat, points.T) + self._shift[:, np.newaxis]).T
