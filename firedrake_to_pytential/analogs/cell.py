import numpy as np

from FIAT.reference_element import Simplex
from modepy import tools

from fd2mm.analogs import Analog
from fd2mm.analogs.utils import get_affine_mapping


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

        # Stored as (dim, nunit_vertices)
        self._unit_vertices = tools.unit_vertices(cell.get_dimension()).T

        # Maps firedrake reference nodes to :mod:`meshmode`
        # unit nodes by x -> Ax + b, where A is :attr:`_mat`
        # and b is :attr:`_shift`
        self._mat, self._shift = get_affine_mapping(cell.vertices.T,
                                                     self._unit_vertices)

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
