"""
    This should only be run with complex firedrake

    Checks to see if I have PETSc set up correctly
"""
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
k = 3

m.init()
V = fd.FunctionSpace(m, 'DG', 1)

# We want this to be just a Laplace
# simple-layer potential with a source on the
# inner boundary (x) evaluated on the outer boundary
# (y)
class MatrixFreeB(object):
    def __init__(self, cl_ctx, queue, function_space,
                 inner_bdy_id, k=3,
                 ambient_dim=None):

        self.queue = queue
        self.k = k

        # {{{ Convert mesh
        self.converter = FiredrakeMeshmodeConnection(
            cl_ctx, queue,
            function_space,
            ambient_dim=ambient_dim,
            source_bdy_id=inner_bdy_id)
        qbx = self.converter.qbx_map['source']
        # }}}

        # {{{  Create operator
        # (This comes from Klockner's pytential/examples/helmholtz-dirichlet.py)

        from sumpy.kernel import HelmholtzKernel
        from pytential import sym, bind

        kernel = HelmholtzKernel(2)

        alpha = 1j

        cse = sym.cse
        sqrt_w = sym.sqrt_jac_q_weight(2)
        density_var = cse(sym.var("sigma")/sqrt_w)

        op = - 0.5 * sym.var("sigma") + \
             (
                alpha*sym.S(kernel, density_var, k=sym.var("k"),
                        qbx_forced_limit=+1) \
                    - sym.D(kernel, density_var, k=sym.var("k"),
                        qbx_forced_limit="avg")
                )

        self.bound_op = bind(qbx, op)

        # }}}

    def mult(self, mat, x, y):
        """
        # Reorder x nodes and put on device
        sigma = self.converter(self.queue, np.array(x))
        sigma = cl.array.to_device(self.queue, sigma)
        sigma = cl.array.to_device(self.queue, np.array(x))
        """
        sigma = cl.array.to_device(self.queue, np.array(x))

        # Perform operation
        y.array = self.bound_op(self.queue, sigma=sigma, k=self.k).get(queue=queue)

from firedrake.petsc import PETSc

# build matrix context
Bctx = MatrixFreeB(cl_ctx, queue, V, inner_bdy_id,k=k,
                   ambient_dim=2)
nnodes = Bctx.converter.qbx_map['source'].density_discr.nnodes
nodes = Bctx.converter.qbx_map['source'].density_discr.nodes().with_queue(queue)

sizes = ((nnodes, nnodes), (nnodes, nnodes))

B = PETSc.Mat().create()


# set up B
B.setSizes(*sizes)

B.setType(B.Type.PYTHON)
B.setPythonContext(Bctx)
B.setUp()

# Now to set up a solver:
ksp = PETSc.KSP().create()

ksp.setOperators(B)

ksp.setFromOptions()

k_vec = np.array([2, 1])
k_vec = k * k_vec / np.linalg.norm(k_vec, 2)

def u_incoming_func(x):
    return cl.clmath.exp(
            1j * (x[0] * k_vec[0] + x[1] * k_vec[1]))

bc = -u_incoming_func(nodes)

from pytential import sym, bind
sqrt_w = sym.sqrt_jac_q_weight(2)
bvp_rhs = bind(Bctx.converter.qbx_map['source'], sqrt_w*sym.var("bc"))(queue, bc=bc).get(queue=queue)

b = PETSc.Vec().create()
b.setSizes(nnodes, nnodes)
b.setFromOptions()

solution = b.duplicate()
b.array = bvp_rhs.copy()

ksp.solve(b, solution)

# Now compute a solution with pytential
from pytential.solve import gmres
gmres_result = gmres(
     Bctx.bound_op.scipy_op(queue, "sigma", dtype=np.complex128, k=k),
        bvp_rhs, tol=1e-8, progress=True,
        stall_iterations=0,
        hard_failure=True,
        maxiter=400)

pyt_sol = gmres_result.solution
print("Max diff=", np.max(np.abs((pyt_sol - solution.array))))
