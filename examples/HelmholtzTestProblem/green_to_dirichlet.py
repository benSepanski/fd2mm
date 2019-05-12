import firedrake as fd

from firedrake_to_pytential.op import fd_bind
from sumpy.kernel import HelmholtzKernel


def green_to_dirichlet(cl_ctx, queue, V, kappa,
                       outer_bdy_id, inner_bdy_id, true_sol_expr,
                       function_converter):
    # away from the excluded region, but firedrake and meshmode point
    # into
    pyt_inner_normal_sign = -1
    ambient_dim = 2
    degree = V.ufl_element().degree()
    m = V.mesh()

    Vdim = fd.VectorFunctionSpace(m, 'CG', degree, dim=ambient_dim)

    # DG Spaces and converters
    V_dg = fd.FunctionSpace(m, 'DG', degree)
    Vdim_dg = fd.VectorFunctionSpace(m, 'DG', degree, dim=ambient_dim)

    # {{{ Create operator
    from pytential import sym

    sigma = sym.make_sym_vector("sigma", ambient_dim)
    op = pyt_inner_normal_sign * (
        sym.D(HelmholtzKernel(2),
              sym.var("u"), k=sym.var("k"),
              qbx_forced_limit=None)
        - sym.S(HelmholtzKernel(2),
                sym.n_dot(sigma), k=sym.var("k"),
                qbx_forced_limit=None)
        )

    pyt_op = fd_bind(function_converter, op, source=(V_dg, inner_bdy_id),
                     target=(V, outer_bdy_id))
    # }}}

    # get true solution
    true_sol = fd.Function(V, name="True Solution").interpolate(true_sol_expr)
    true_sol_grad = fd.Function(Vdim).interpolate(fd.grad(true_sol_expr))

    true_sol_dg = fd.project(true_sol, V_dg, use_slate_for_inverse=False)
    true_sol_grad_dg = fd.project(true_sol_grad, Vdim_dg,
                                  use_slate_for_inverse=False)

    outer_bdy_condition = pyt_op(
        queue, u=true_sol_dg, sigma=true_sol_grad_dg, k=kappa)

    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)

    a = (fd.inner(fd.grad(u), fd.grad(v)) - kappa**2 * fd.inner(u, v)) * fd.dx
    L = fd.inner(fd.inner(true_sol_grad, fd.FacetNormal(V.mesh())),
                 v) * fd.ds(inner_bdy_id)
    bc = fd.DirichletBC(V, outer_bdy_condition, outer_bdy_id)

    solution = fd.Function(V)
    fd.solve(a == L, solution, bcs=[bc])

    return solution
