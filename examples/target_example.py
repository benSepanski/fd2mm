import pyopencl as cl
import pyopencl.clmath

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

import firedrake as fd
import numpy as np

from firedrake_to_pytential import FiredrakeMeshmodeConnection

m = fd.Mesh('square_ring.msh')
outer_bdy_id = 1
inner_bdy_id = 2

m.init()
V = fd.FunctionSpace(m, 'DG', 1)
converter = FiredrakeMeshmodeConnection(cl_ctx, V, ambient_dim=2,
                                        source_bdy_id=inner_bdy_id,
                                        target_bdy_id=outer_bdy_id)
py_mesh = converter.mesh_map['source']

xx = fd.SpatialCoordinate(m)
f = fd.Function(V).interpolate(fd.sin(xx[0]))
fntn = converter(queue, f)
fntn = cl.array.to_device(queue, fntn)

from pytential import bind, sym
alpha = 1j
k = 3
sigma_sym = sym.var("sigma")

from sumpy.kernel import HelmholtzKernel
kernel = HelmholtzKernel(2)
cse = sym.cse
sigma_sym = sym.var("sigma")
loc_sign = +1
op = ( alpha*sym.S(kernel, sigma_sym, k=sym.var("k"),
            qbx_forced_limit=None)
        -sym.D(kernel, sigma_sym, k=sym.var("k"),
            qbx_forced_limit=None)
      )

bound_op = bind((converter.qbx_map['source'], converter.target_points), op)
eval_potential = bound_op(queue, sigma=fntn, k=k)
