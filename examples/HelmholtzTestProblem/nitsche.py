import firedrake as fd

from firedrake_to_pytential.op import fd_bind, FunctionConverter
from sumpy.kernel import HelmholtzKernel


def nitsche(cl_ctx, queue, V, kappa,
            outer_bdy_id, inner_bdy_id, true_sol_expr,
            function_converter):
    # away from the excluded region, but firedrake and meshmode point
    # into
    pyt_inner_normal_sign = -1
    ambient_dim = 2
    degree = V.ufl_element().degree()
    m = V.mesh()

    # nitsche parameters
    gamma = 10
    h = fd.CellSize(m)

    Vdim = fd.VectorFunctionSpace(m, 'CG', degree, dim=ambient_dim)

    # DG Spaces and converters
    V_dg = fd.FunctionSpace(m, 'DG', degree)
    Vdim_dg = fd.VectorFunctionSpace(m, 'DG', degree, dim=ambient_dim)

    # {{{ Create operator
    from pytential import sym

    op = pyt_inner_normal_sign * sym.D(HelmholtzKernel(dim=2),
                                       sym.var("u"), k=sym.var("k"),
                                       qbx_forced_limit=None)

    pyt_mat_op = fd_bind(function_converter, op, source=(V_dg, inner_bdy_id),
                         target=(V, outer_bdy_id))
    # }}}

    class MatrixFreeB(object):
        def __init__(self, A, pyt_op, queue, kappa):
            """
            :arg kappa: The wave number
            """

            self.queue = queue
            self.k = kappa
            self.pyt_op = pyt_op

            # {{{ Create some functions needed for multing
            self.x_fntn = fd.Function(V)
            self.x_dg_fntn = fd.Function(V_dg)
            self.potential_int = fd.Function(V)
            self.potential_int.dat.data[:] = 0.0
            self.v = fd.TestFunction(V)
            self.n = fd.FacetNormal(m)
            # }}}

        def mult(self, mat, x, y):
            # Perform pytential operation
            self.x_fntn.dat.data[:] = x[:]
            self.x_dg_fntn = fd.project(self.x_fntn, V_dg,
                                        use_slate_for_inverse=False)
            # potential_int will hold
            # u *_\Gamma \partial_n G
            self.pyt_op(self.queue, result_function=self.potential_int,
                        u=self.x_dg_fntn, k=self.k)

            """
                compute

                .. math::

                (u *_\Gamma \partial_n G, \partial_n v)_\Sigma
                - gamma / h (u *_\Gamma \partial_n G, v)_\Sigma
            """
            self.potential_int = fd.assemble(
                (fd.inner(self.potential_int,
                          fd.inner(fd.grad(self.v), self.n))
                 - gamma / h * fd.inner(self.potential_int, self.v))
                * fd.ds(outer_bdy_id))

            # y <- Ax + evaluated potential
            A.mult(x, y)
            with self.potential_int.dat.vec_ro as ep:
                y.axpy(+1, ep)

    from firedrake.petsc import PETSc

    # {{{ Compute normal helmholtz operator
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)

    r"""
        This is the part we can assemple on :mod:`firedrake`

        .. math::

            (\nabla u, \nabla v)
            - \kappa^2 (\u, v)
            - (\partial_n u, v)_\Sigma
            - (u, \partial_n v)_\Sigma
            + \gamma / h (u, v)_\Sigma

    """
    u_n = fd.inner(fd.grad(u), fd.FacetNormal(m))
    v_n = fd.inner(fd.grad(v), fd.FacetNormal(m))
    a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx \
        - kappa**2 * fd.inner(u, v) * fd.dx \
        - fd.inner(u_n, v) * fd.ds(outer_bdy_id) \
        - fd.inner(u, v_n) * fd.ds(outer_bdy_id) \
        + gamma / h * fd.inner(u, v) * fd.ds(outer_bdy_id)
    # get the concrete matrix from a general bilinear form
    A = fd.assemble(a).M.handle
    # }}}

    # {{{ Setup Python matrix
    B = PETSc.Mat().create()

    # build matrix context
    Bctx = MatrixFreeB(A, pyt_mat_op, queue, kappa)

    # set up B as same size as A
    B.setSizes(*A.getSizes())

    B.setType(B.Type.PYTHON)
    B.setPythonContext(Bctx)
    B.setUp()
    # }}}

    # {{{ Create rhs

    # get true solution
    true_sol = fd.Function(V, name="True Solution").interpolate(true_sol_expr)
    true_sol_grad = fd.Function(Vdim).interpolate(fd.grad(true_sol_expr))

    # Remember f is \partial_n(true_sol)|_\Gamma
    # so we just need to compute \int_\Gamma\partial_n(true_sol) H(x-y)
    from pytential import sym

    sigma = sym.make_sym_vector("sigma", ambient_dim)
    op = pyt_inner_normal_sign * sym.S(HelmholtzKernel(dim=ambient_dim),
                                       sym.n_dot(sigma),
                                       k=sym.var("k"), qbx_forced_limit=None)

    rhs_op = fd_bind(function_converter, op, source=(Vdim_dg, inner_bdy_id),
                     target=(V, outer_bdy_id))

    true_sol_grad_dg = fd.project(true_sol_grad, Vdim_dg,
                                  use_slate_for_inverse=False)
    f_convoluted = rhs_op(queue, sigma=true_sol_grad_dg, k=kappa)
    """
    ..math::

        (f, v)_\Gamma
        + (f * G, \partial_n v)_\Sigma
        - \gamma / h (f * G, v)_\Sigma

    """
    # Recall that v_n is \partial_n v
    f = fd.inner(fd.grad(true_sol), fd.FacetNormal(m))
    rhs_form = fd.inner(f, v) * fd.ds(inner_bdy_id) \
        + fd.inner(f_convoluted, v_n) * fd.ds(outer_bdy_id) \
        - gamma / h * fd.inner(f_convoluted, v) * fd.ds(outer_bdy_id)

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

    return solution
