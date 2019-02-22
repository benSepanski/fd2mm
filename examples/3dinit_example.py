import pyopencl as cl

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

import firedrake as fd
from firedrake_to_pytential import FiredrakeMeshmodeConnection

m = fd.UnitCubeMesh(10, 10, 10)
#m = fd.UnitOctahedralSphereMesh(refinement_level=2)
#m.init_cell_orientations(fd.SpatialCoordinate(m))
V = fd.FunctionSpace(m, 'DG', 1)

converter = FiredrakeMeshmodeConnection(cl_ctx, V)
py_mesh = converter.mesh_map['source']

xx = fd.SpatialCoordinate(m)
f = fd.Function(V).interpolate(fd.sin(xx[0]))
target = converter(None, f)

from meshmode.mesh import check_bc_coverage
