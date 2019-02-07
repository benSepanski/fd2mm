import pyopencl as cl

cl_ctx = cl.create_some_context()

import firedrake as fd
from firedrake_to_pytential import FiredrakeMeshmodeConnection

m = fd.Mesh('square_ring.msh')
m.init()
V = fd.FunctionSpace(m, 'DG', 1)
converter = FiredrakeMeshmodeConnection(cl_ctx, V, ambient_dim=2)
py_mesh = converter.mesh_map['source']

xx = fd.SpatialCoordinate(m)
f = fd.Function(V).interpolate(fd.sin(xx[0]))
target = converter(None, f)

from meshmode.mesh import check_bc_coverage
check_bc_coverage(py_mesh, [1, 2])
