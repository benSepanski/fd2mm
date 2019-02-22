import pyopencl as cl
import pyopencl.clmath

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

import firedrake as fd
import numpy as np

from firedrake_to_pytential import FiredrakeMeshmodeConnection

omega = 250
c = 340
kappa = omega / c

# for the double layer formulation, the normal points
# away from the excluded region, but firedrake and meshmode point
# into
inner_normal_sign = -1
#m = fd.Mesh('square_ring.msh')
m = fd.Mesh('annulus.msh')
outer_bdy_id = 1
inner_bdy_id = 2

m.init()

degree = 1
ambient_dim = 2
V = fd.FunctionSpace(m, 'CG', degree)
Vdim = fd.VectorFunctionSpace(m, 'CG', degree, dim=ambient_dim)
V_dg = fd.FunctionSpace(m, 'DG', degree)
Vdim_dg = fd.VectorFunctionSpace(m, 'DG', degree, dim=ambient_dim)

converter = FiredrakeMeshmodeConnection(
    cl_ctx, V_dg,
    ambient_dim=ambient_dim,
    source_bdy_id=inner_bdy_id)

grad_converter = FiredrakeMeshmodeConnection(
    cl_ctx, Vdim_dg,
    ambient_dim=ambient_dim,
    source_bdy_id=inner_bdy_id)

class MatrixFreeB(object):
    def __init__(self, A, converter, queue, kappa, targets, target_indices):
        """
        :arg kappa: The wave number
        :arg targets: Locations to compute grad at
        """

        self.queue = queue
        self.converter = converter
        self.k = kappa
        self.target_indices = target_indices
        self.directions = np.copy(targets)
        for i in range(targets.shape[1]):
            norm = np.linalg.norm(self.directions[:, i])
            self.directions[:, i] /= norm
        self.directions = cl.array.to_device(self.queue, self.directions)

        # {{{  Create operator
        from sumpy.kernel import HelmholtzKernel
        from pytential import sym, bind

        """
            (x/|x|\cdot \nabla-i\kappa)\left(
                \int_\Gamma \partial_{n(y)}H(x-y)u(y) d\gamma(y)
            \right)

            i.e. \nabla_x\left(\partial_nH *_\Gamma u\right)
        """
        directions = sym.make_sym_vector("directions", ambient_dim)
        op = np.dot(
                directions,
                sym.grad(ambient_dim,
                sym.D(HelmholtzKernel(2), sym.var("u"), k=sym.var("k"),
                      qbx_forced_limit=None)
                )
            ) - 1j * sym.var("k") * \
            sym.D(HelmholtzKernel(2), sym.var("u"), k=sym.var("k"),
                  qbx_forced_limit=None)
        # }}}

        # {{{ Bind operator
        from pytential.target import PointsTarget

        qbx = self.converter.qbx_map['source']
        self.bound_op = bind((qbx, PointsTarget(targets)), op)
        # }}}

    def mult(self, mat, x, y):
        # Reorder x nodes and put on device
        x_fntn = fd.Function(V)
        x_fntn.dat.data[:] = x.array[:]
        x_fntn = fd.project(x_fntn, V_dg)
        u = self.converter(self.queue, x_fntn)
        u = cl.array.to_device(self.queue, u)

        # Perform operation
        eval_potential = self.bound_op(self.queue, u=u, k=self.k, directions=self.directions)
        eval_potential = eval_potential.get(queue=queue)

        potential_int = fd.Function(V)
        potential_int.dat.data[self.target_indices] = eval_potential
        v = fd.TestFunction(V)
        """
            \int_\Sigma 
                (x/|x|\cdot\nabla - i\kappa)
                left(
                    \int_\Gamma \partial_{n(y)}H(x-y)u(y) d\gamma(y)
                \right) * v
            d\sigma(x)

            i.e. \langle \partial_{n(x)}\left(\partial_nH*_\Gamma u\right), v \rangle_\Sigma
        """
        potential_int = fd.assemble(fd.inner(potential_int, v) * fd.ds(outer_bdy_id))

        # y <- Ax - evaluated potential
        A.mult(x, y)
        with potential_int.dat.vec_ro as ep:
            y.axpy(-inner_normal_sign, ep)

# {{{ get targets, target_indices
target_indices = V.boundary_nodes(outer_bdy_id, 'topological')

xx = fd.SpatialCoordinate(m)
coords = fd.Function(Vdim).interpolate(xx)
targets = np.array(coords.dat.data[target_indices], dtype=np.float64)
# Change from [nnodes][ambient_dim] to [ambient_dim][nnodes]
targets = np.transpose(targets).copy()

# }}} 

from firedrake.petsc import PETSc

# {{{ Compute normal helmholtz operator
u = fd.TrialFunction(V)
v = fd.TestFunction(V)
a = (fd.inner(fd.grad(u), fd.grad(v)) - kappa**2 * fd.inner(u, v)) * fd.dx \
    - 1j * kappa * fd.inner(u, v) * fd.ds(outer_bdy_id)
# get the concrete matrix from a general bilinear form
A = fd.assemble(a).M.handle
# }}}

# {{{ Setup Python matrix
B = PETSc.Mat().create()

# build matrix context
Bctx = MatrixFreeB(A, converter, queue, kappa, targets, target_indices)

# set up B as same size as A
B.setSizes(*A.getSizes())

B.setType(B.Type.PYTHON)
B.setPythonContext(Bctx)
B.setUp()
# }}}

# {{{ Create rhs

# get true solution
from pml_functions import hankel_function
true_sol_expr = fd.Constant(1j / 4) * \
    hankel_function(kappa * fd.sqrt((xx[0] - 0.5)**2 + xx[1]**2))
"""
true_sol_expr = hankel_function(kappa * fd.sqrt(xx[0]**2 + xx[1]**2))
"""
true_sol = fd.Function(V, name="True Solution").interpolate(true_sol_expr)
true_sol_grad = fd.Function(Vdim).interpolate(fd.grad(true_sol_expr))

# Remember f is \partial_n(true_sol)|_\Gamma
# so we just need to compute \int_\Gamma\partial_n(true_sol) H(x-y)
from pytential import sym, bind
from pytential.target import PointsTarget
from sumpy.kernel import HelmholtzKernel

"""
    For a vector function sigma

    (x/|x|\cdot\nabla-i * \kappa) \left(
        \int_\Gamma H(x-y) n(y)\cdot sigma(y) d\gamma(y)
    \right)
"""
sigma = sym.make_sym_vector("sigma", ambient_dim)
directions = sym.make_sym_vector("directions", ambient_dim)
op = np.dot(
    directions,
    sym.grad(ambient_dim, sym.S(HelmholtzKernel(ambient_dim),
           sym.n_dot(sigma),
           k=sym.var("k"),
           qbx_forced_limit=None))) - 1j * sym.var("k") * \
    sym.S(HelmholtzKernel(ambient_dim), sym.n_dot(sigma),
          k=sym.var("k"), qbx_forced_limit=None)

qbx = converter.qbx_map['source']
bound_op = bind((qbx, PointsTarget(targets)), op)

true_sol_grad_dg = fd.project(true_sol_grad, Vdim_dg)
true_sol_grad_pyt = cl.array.to_device(queue, grad_converter(queue, true_sol_grad_dg))
f_convo = bound_op(queue, sigma=true_sol_grad_pyt, k=kappa,
                   directions=Bctx.directions)
f_convo = f_convo.get(queue=queue)

f_convoluted = fd.Function(V)
for i, ind in enumerate(target_indices):
    f_convoluted.dat.data[ind] = f_convo[i]

"""
\langle \partial_n true_sol, v\rangle_\Gamma
 - \langle x/|x|\cdot (H*\partial_n true_sol), v\rangle_\Sigma
"""
rhs_form = fd.inner(
                fd.inner(fd.grad(true_sol), fd.FacetNormal(m)),
                v) * fd.ds(inner_bdy_id) \
        - inner_normal_sign * fd.inner(f_convoluted, v) * fd.ds(outer_bdy_id)

rhs = fd.assemble(rhs_form)

# {{{ set up a solver:
solution = fd.Function(V, name="Computed Solution")

ksp = PETSc.KSP().create()
# precondition with A
ksp.setOperators(B, A)

ksp.setFromOptions()
with rhs.dat.vec_ro as b:
    with solution.dat.vec as x:
        ksp.solve(b, x)
# }}}

# {{{ Evaluate accuracy
error = fd.Function(V, name="Error").interpolate(true_sol - solution)
neg_error = fd.Function(V).interpolate(true_sol + solution)

l2error = fd.sqrt(fd.assemble(fd.inner(error, error) * fd.dx))
l2neg_error = fd.sqrt(fd.assemble(fd.inner(neg_error, neg_error) * fd.dx))

if l2neg_error < l2error:
    raise UserWarning("""-(computed solution) performs better than computed solution,
off by a sign?""")

true_sol_norm = fd.sqrt(fd.assemble(fd.inner(true_sol, true_sol) * fd.dx))
rel_error = l2error / true_sol_norm

ptwise_rel_error = fd.Function(V, name="Relative Error")
ptwise_rel_error.interpolate(fd.sqrt(fd.inner(error, error) / fd.inner(true_sol, true_sol)))

print("\\kappa=", kappa)
print("L^2 error:", l2error)
print("Relative error:", round(rel_error * 100, 2), "%")

import matplotlib.pyplot as plt
#fd.plot(solution)
#fd.plot(true_sol)
#fd.plot(error)
#fd.plot(ptwise_rel_error, real=True)
plt.show()
