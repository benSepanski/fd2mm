"""
    May need to add this package to PYTHONPATH for it to run:
PYTHONPATH=$PYTHONPATH:<parent directory of whatever folder the package is in>
"""
import pyopencl as cl
import pyopencl.clmath

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

import firedrake as fd
import numpy as np

from firedrake_to_pytential import FiredrakeToMeshmodeConverter

m = fd.Mesh('square_ring.msh')
outer_bdy_id = 1
inner_bdy_id = 2

m.init()
V = fd.FunctionSpace(m, 'DG', 1)
converter = FiredrakeToMeshmodeConverter(V, ambient_dim=2)
py_mesh = converter.get_meshmode_mesh()

xx = fd.SpatialCoordinate(m)
f = fd.Function(V).interpolate(fd.sin(xx[0]))
fntn = converter.fd_function_to_array(f)

from pytential import bind, sym
alpha = 1j
k = 3
sigma_sym = sym.var("sigma")

from sumpy.kernel import HelmholtzKernel
kernel = HelmholtzKernel(2)
cse = sym.cse
sigma_sym = sym.var("sigma")
sqrt_w = sym.sqrt_jac_q_weight(2)
inv_sqrt_w_sigma = cse(sigma_sym/sqrt_w)
loc_sign = +1
op = (-loc_sign*0.5*sigma_sym
    + sqrt_w * (
        alpha*sym.S(kernel, inv_sqrt_w_sigma, k=sym.var("k"),
            qbx_forced_limit=+1)
        -sym.D(kernel, inv_sqrt_w_sigma, k=sym.var("k"),
            qbx_forced_limit="avg")
        ))

qbx = converter.get_qbx(cl_ctx)
density_discr = qbx.density_discr


# Get bdy connection
from meshmode.discretization.connection import \
    make_face_restriction
from meshmode.discretization.poly_element import \
    InterpolatoryQuadratureSimplexGroupFactory

inner_bdy_connection = make_face_restriction(
    density_discr, InterpolatoryQuadratureSimplexGroupFactory(1), inner_bdy_id)
inner_bdy_mesh = inner_bdy_connection.to_discr.mesh
from pytential.qbx import QBXLayerPotentialSource
inner_bdy_qbx = QBXLayerPotentialSource(
                inner_bdy_connection.to_discr,
                fine_order=1, qbx_order=1, fmm_order=1)
density_discr = inner_bdy_qbx.density_discr

bound_op = bind(inner_bdy_qbx, op)
fntn = cl.array.to_device(queue, fntn)
fntn_on_inner_bdy = inner_bdy_connection(queue, fntn).with_queue(queue)

k_vec = np.array([2, 1])
k_vec = k * k_vec / np.linalg.norm(k_vec, 2)

def u_incoming_func(x):
    return cl.clmath.exp(
            1j * (x[0] * k_vec[0] + x[1] * k_vec[1]))

nodes = density_discr.nodes().with_queue(queue)
bc = -u_incoming_func(nodes)

bvp_rhs = bind(inner_bdy_qbx, sqrt_w*sym.var("bc"))(queue, bc=bc)

from pytential.solve import gmres
gmres_result = gmres(
        bound_op.scipy_op(queue, "sigma", dtype=np.complex128, k=k),
        bvp_rhs, tol=1e-8, progress=True,
        stall_iterations=0,
        hard_failure=True)

bound_op(queue, sigma=fntn_on_inner_bdy, k=k)
