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

from sumpy.kernel import HelmholtzKernel
kernel = HelmholtzKernel(2)
cse = sym.cse

op = alpha*sym.S(kernel,sym.var("sigma"), k=sym.var("k"),
            qbx_forced_limit=None) \
        - sym.D(kernel, sym.var("sigma"), k=sym.var("k"),
            qbx_forced_limit=None)

qbx = converter.get_qbx(cl_ctx)
density_discr = qbx.density_discr


# Get bdy connections
from meshmode.discretization.connection import \
    make_face_restriction
from meshmode.discretization.poly_element import \
    InterpolatoryQuadratureSimplexGroupFactory

outer_bdy_connection = make_face_restriction(
    density_discr, InterpolatoryQuadratureSimplexGroupFactory(1), outer_bdy_id)
from pytential.target import PointsTarget
target = outer_bdy_connection.to_discr.nodes().with_queue(queue)

inner_bdy_connection = make_face_restriction(
    density_discr, InterpolatoryQuadratureSimplexGroupFactory(1), inner_bdy_id)
inner_bdy_mesh = inner_bdy_connection.to_discr.mesh
from pytential.qbx import QBXLayerPotentialSource
inner_bdy_qbx = QBXLayerPotentialSource(
                inner_bdy_connection.to_discr,
                fine_order=1, qbx_order=1, fmm_order=1)
density_discr = inner_bdy_qbx.density_discr

bound_op = bind((inner_bdy_qbx, PointsTarget(target)), op)
fntn = cl.array.to_device(queue, fntn)
fntn_on_inner_bdy = inner_bdy_connection(queue, fntn).with_queue(queue)

bound_op(queue, sigma=fntn_on_inner_bdy, k=k)
