import pyopencl as cl

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

from random import randint
import numpy as np
import matplotlib.pyplot as plt

from firedrake import Mesh, FunctionSpace, VectorFunctionSpace, \
    Function, FacetNormal, ds, assemble, inner, sqrt, plot

from firedrake_to_pytential.op import fd_bind, FunctionConverter
from sumpy.kernel import HelmholtzKernel

# Problem setup
num_iter = 5
c = 340

degree = 1
fmm_order = 10  # Determines accuracy of potential evaluation
with_refinement = False
mesh = Mesh("meshes/domain.msh")

fine_order = degree
qbx_order = degree

pml_x_region = 1
pml_y_region = 2
pml_xy_region = 3
inner_region = 4
outer_bdy_id = 6
inner_bdy_id = 5

V = FunctionSpace(mesh, 'CG', degree)
function_converter = FunctionConverter(cl_ctx,
                                       fine_order=fine_order,
                                       fmm_order=fmm_order,
                                       qbx_order=qbx_order,
                                       with_refinement=with_refinement)

# away from the excluded region, but firedrake and meshmode point
# into

pyt_inner_normal_sign = -1
ambient_dim = 2
degree = V.ufl_element().degree()
m = V.mesh()

Vdim = VectorFunctionSpace(m, 'CG', degree, dim=ambient_dim)

# {{{ Create operator
from pytential import sym

"""
..math:

x \in \Sigma

grad_op(x) =
    \nabla(
        \int_\Gamma(
            u(y) \partial_n H_0^{(1)}(\kappa |x - y|)
        )d\gamma(y)
    )
"""
grad_op = pyt_inner_normal_sign * sym.grad(
    2, sym.D(HelmholtzKernel(dim=2),
             sym.var("u"), k=sym.var("k"),
             qbx_forced_limit=None)
             )

"""
..math:

x \in \Sigma

op(x) =
    i \kappa \cdot
    \int_\Gamma(
        u(y) \partial_n H_0^{(1)}(\kappa |x - y|)
    )d\gamma(y)
"""
op = pyt_inner_normal_sign * 1j * sym.var("k") * (
    sym.D(HelmholtzKernel(2),
            sym.var("u"), k=sym.var("k"),
            qbx_forced_limit=None)
    )

# The operation we want is -(grad_op\cdot normal - op)

pyt_grad_op = fd_bind(function_converter, grad_op, source=(V, inner_bdy_id),
                      target=(Vdim, outer_bdy_id))
pyt_op = fd_bind(function_converter, op, source=(V, inner_bdy_id),
                     target=(V, outer_bdy_id))

n = FacetNormal(mesh)

rand_fntn = Function(V)
fntn_shape = rand_fntn.dat.data.shape

result_fntn_dim = Function(Vdim)
result_fntn = Function(V)
result_fntn_dim.dat.data[:] = 0
result_fntn.dat.data[:] = 0

kappas = []
results = []

for i in range(num_iter):
    omega = randint(c // 10, c * 20)
    kappa = omega / c

    norm = 0.0
    while norm < 0.01:
        rand_fntn.dat.data[:] = np.random.rand(*fntn_shape)[:]
        # Normalize on inner boundary
        norm = sqrt(assemble(
            inner(rand_fntn, rand_fntn) * ds(inner_bdy_id)
            ))
    rand_fntn.dat.data[:] = rand_fntn.dat.data[:] / norm

    # Compute pytential operations
    pyt_op(queue, result_function=result_fntn,
           u=rand_fntn, k=kappa)
    pyt_grad_op(queue, result_function=result_fntn_dim,
                u=rand_fntn, k=kappa)

    # (Ku, u)_{outer bdy}
    result = assemble(
        - inner(inner(result_fntn_dim, n) - result_fntn,
                rand_fntn) * ds(outer_bdy_id)
        )

    kappas.append(kappa)
    results.append(result)

    if i > 0 and i % 100 == 0:
        print("Iter %s/%s Complete" % (i, num_iter))

# }}}

plt.scatter(kappas, results)
plt.title("Guaranteed Accuracy = %s" % (2 ** -fmm_order))
plt.xlabel("Wave Number")
plt.ylabel("Norm of Result")
plt.show()

out = open("out.txt", 'w')
for kap, res in zip(kappas, results):
    out.write("%s %s\n" % (kap, res))

out.close()
