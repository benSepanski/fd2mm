import pyopencl as cl
import pyopencl.clmath

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

import firedrake as fd
import numpy as np

from firedrake_to_pytential import FiredrakeMeshmodeConnection

m = fd.Mesh('m.msh')
m.init()
V = fd.FunctionSpace(m, 'DG', 1)
converter = FiredrakeMeshmodeConnection(cl_ctx, V, ambient_dim=3)
py_mesh = converter.mesh_map['source']

xx = fd.SpatialCoordinate(m)
f = fd.Function(V).interpolate(fd.sin(xx[0]))
fntn = converter(queue, f)

from pytential import bind, sym
alpha = 1j
k = 3
sigma_sym = sym.var("sigma")

from sumpy.kernel import HelmholtzKernel
kernel = HelmholtzKernel(3)
cse = sym.cse
sigma_sym = sym.var("sigma")
sqrt_w = sym.sqrt_jac_q_weight(3)
inv_sqrt_w_sigma = cse(sigma_sym/sqrt_w)
loc_sign = +1
op = (-loc_sign*0.5*sigma_sym
    + sqrt_w * (
        alpha*sym.S(kernel, inv_sqrt_w_sigma, k=sym.var("k"),
            qbx_forced_limit=+1)
        -sym.D(kernel, inv_sqrt_w_sigma, k=sym.var("k"),
            qbx_forced_limit="avg")
        ))

qbx = converter.qbx_map['source']

bound_op = bind(qbx, op)
fntn = cl.array.to_device(queue, fntn)

bound_op(queue, sigma=fntn, k=k)
