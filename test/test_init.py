"""
    May need to add this package to PYTHONPATH for it to run:
PYTHONPATH=$PYTHONPATH:<parent directory of whatever folder the package is in>
"""

import firedrake as fd
from firedrake_to_pytential import FiredrakeToMeshmodeConverter

m = fd.Mesh('square_ring.msh')
m.init()
V = fd.FunctionSpace(m, 'DG', 1)
converter = FiredrakeToMeshmodeConverter(V, ambient_dim=2)
py_mesh = converter.get_meshmode_mesh()

xx = fd.SpatialCoordinate(m)
f = fd.Function(V).interpolate(fd.sin(xx[0]))
target = converter.fd_function_to_array(f)

from meshmode.mesh import check_bc_coverage
check_bc_coverage(py_mesh, [1,2])
