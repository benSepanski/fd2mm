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

# We want this to be just a Laplace
# simple-layer potential with a source on the
# inner boundary (x) evaluated on the outer boundary
# (y)
class MatrixFreeB(object):
    def __init__(self, cl_ctx, queue, function_space,
                 outer_bdy_id, inner_bdy_id,
                 ambient_dim=None):

        self.queue = queue

        # {{{ Convert mesh
        self.converter = FiredrakeMeshmodeConnection(
            cl_ctx, queue,
            function_space,
            ambient_dim=ambient_dim,
            source_bdy_id=inner_bdy_id,
            target_bdy_id=outer_bdy_id)
        qbx = self.converter.qbx_map['source']
        # }}}

        # {{{  Create operator
        from sumpy.kernel import LaplaceKernel
        from pytential import sym, bind
        op = sym.S(LaplaceKernel(2), sym.var("sigma"),
                   qbx_forced_limit=None)

        from pytential.target import PointsTarget
        targets = self.converter.target_points.nodes().with_queue(queue)
        self.bound_op = bind((qbx, PointsTarget(targets)), op)
        # }}}

    def mult(self, mat, x, y):
        # Reorder x nodes and put on device
        sigma = self.converter(self.queue, np.array(x))
        sigma = cl.array.to_device(self.queue, sigma)

        # Perform operation
        evaluated_potential = self.bound_op(self.queue, sigma=sigma)

        # Convert to a firedrake node ordering
        evaluated_potential = self.converter(queue, evaluated_potential, invert=True)

        # y <- evaluated_potential
        y[:] = evaluated_potential[:]

from firedrake.petsc import PETSc

u = fd.TrialFunction(V)
v = fd.TestFunction(V)
a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
# get the concrete matrix from a general bilinear form
A = fd.assemble(a).M.handle

B = PETSc.Mat().create()

# build matrix context
Bctx = MatrixFreeB(cl_ctx, queue, V, outer_bdy_id, inner_bdy_id,
                   ambient_dim=2)

# set up B as same size as A
B.setSizes(*A.getSizes())

B.setType(B.Type.PYTHON)
B.setPythonContext(Bctx)
B.setUp()

# Now to set up a solver:
ksp = PETSc.KSP().create()

ksp.setOperators(B)

ksp.setFromOptions()

xx = fd.SpatialCoordinate(m)
f = fd.Function(V).interpolate(fd.sin(xx[0]))
rhs_form = fd.inner(f, v) * fd.dx

# Now compute a solution
solution = fd.Function(V)
rhs = fd.assemble(rhs_form)

with rhs.dat.vec_ro as b:
    with solution.dat.vec as x:
        B.mult(x, b)
