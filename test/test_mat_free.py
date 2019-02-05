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
        self.converter = FiredrakeToMeshmodeConverter(
            function_space,
            ambient_dim=ambient_dim)
        qbx = self.converter.get_qbx(cl_ctx)
        density_discr = qbx.density_discr
        # }}}

        # {{{ Get connections to/from boundary
        from meshmode.discretization.connection import \
            make_face_restriction
        from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

        #  'outer_bdy' -> outer_bdy connection
        #  'inner_bdy' -> inner_bdy connection
        self.connections = {}
        self.connections['outer_bdy'] = make_face_restriction(
            density_discr,
            InterpolatoryQuadratureSimplexGroupFactory(1),
            outer_bdy_id)
        self.connections['inner_bdy'] = make_face_restriction(
            density_discr,
            InterpolatoryQuadratureSimplexGroupFactory(1),
            inner_bdy_id)

        """
        from meshmode.discretization.connection import \
            FACE_RESTR_ALL, make_face_to_all_faces_embedding
        all_faces_connection = make_face_restriction(
            density_discr,
            InterpolatoryQuadratureSimplexGroupFactory(1),
            FACE_RESTR_ALL)
        
        # create reverse connections
        self.reverse_connections = {}
        for (bdy, connection) in self.connections.items():
            self.reverse_connections[bdy] = \
                make_face_to_all_faces_embedding(connection, all_faces_connection.to_discr)
        """

        # }}}

        # {{{  Create operator
        from sumpy.kernel import LaplaceKernel
        from pytential import sym, bind
        op = sym.S(LaplaceKernel(2), sym.var("sigma"),
                   qbx_forced_limit=None)

        from pytential.qbx import QBXLayerPotentialSource
        inner_bdy_qbx = QBXLayerPotentialSource(
            self.connections['inner_bdy'].to_discr,
            fine_order=1, qbx_order=1, fmm_order=1)

        from pytential.target import PointsTarget
        targets = self.connections['outer_bdy'].to_discr.nodes().with_queue(
            self.queue)
        self.bound_op = bind((inner_bdy_qbx, PointsTarget(targets)), op)
        # }}}

    def mult(self, mat, x, y):
        # Reorder x nodes and put on device
        x_array = np.array(x)
        self.converter.reorder_nodes(x_array)
        sigma = cl.array.to_device(self.queue, x_array)

        # Project x onto the inner boundary
        sigma = self.connections['inner_bdy'](self.queue, sigma).with_queue(
            self.queue)

        # Perform operation
        evaluated_potential = self.bound_op(self.queue, sigma=sigma)
        1/0

        # Rewrite evaluated_potential as a function on the whole mesh
        # rather than just the outer boundary
        evaluated_potential = self.reverse_connections['outer_bdy'](
            self.queue, evaluated_potential)

        # Convert to a firedrake function
        evaluated_potential = self.converter.array_to_fd_function(
                                  evaluated_potential)

        # y <- evaluated_potential
        y.dat.data[:] = evaluated_potential.dat.data[:]

from firedrake.petsc import PETSc

u = fd.TrialFunction(V)
v = fd.TestFunction(V)
a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
# get the concrete matrix from a general bilinear form
A = fd.assemble(a).M.handle

B = PETSc.Mat().create()

# build matrix context
Bctx = MatrixFreeB(cl_ctx, queue, V, outer_bdy_id, inner_bdy_id)

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
        ksp.solve(b, x)
