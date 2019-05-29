import firedrake as fd
from firedrake.petsc import PETSc

from firedrake_to_pytential.op import fd_bind
from sumpy.kernel import HelmholtzKernel


def gmati_coupling(cl_ctx, queue, V, kappa,
                   outer_bdy_id, inner_bdy_id, true_sol_expr,
                   function_converter):
    # away from the excluded region, but firedrake and meshmode point
    # into
    rhs_comp = PETSc.Log.Stage("RHS_Comp")
    firedrake_op = PETSc.Log.Stage("Firedrake Comp")
    create_pyt_ops = PETSc.Log.Stage("Make Pyt Op")
    conversion = PETSc.Log.Stage("Conversion")
    create_pyt_ops.push()

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

    pyt_grad_op = fd_bind(function_converter, grad_op, source=(V_dg, inner_bdy_id),
                          target=(Vdim, outer_bdy_id))
    pyt_op = fd_bind(function_converter, op, source=(V_dg, inner_bdy_id),
                         target=(V, outer_bdy_id))
    create_pyt_ops.pop()
    # }}}

    class MatrixFreeB(object):
        def __init__(self, A, pyt_grad_op, pyt_op, queue, kappa):
            """
            :arg kappa: The wave number
            """

            self.queue = queue
            self.k = kappa
            self.pyt_op = pyt_op
            self.pyt_grad_op = pyt_grad_op
            self.A = A

            # {{{ Create some functions needed for multing
            self.x_fntn = fd.Function(V)
            self.x_dg_fntn = fd.Function(V_dg)

            self.potential_int = fd.Function(V)
            self.potential_int.dat.data[:] = 0.0
            self.grad_potential_int = fd.Function(Vdim)
            self.grad_potential_int.dat.data[:] = 0.0
            self.pyt_result = fd.Function(V)

            self.n = fd.FacetNormal(m)
            self.v = fd.TestFunction(V)
            # }}}

        def mult(self, mat, x, y):
            # Perform pytential operation
            conversion.push()
            self.x_fntn.dat.data[:] = x[:]
            self.x_dg_fntn = fd.project(self.x_fntn, V_dg,
                                        use_slate_for_inverse=False)
            self.pyt_op(self.queue, result_function=self.potential_int,
                        u=self.x_dg_fntn, k=self.k)
            self.pyt_grad_op(self.queue, result_function=self.grad_potential_int,
                             u=self.x_dg_fntn, k=self.k)
            conversion.pop()

            # Integrate the potential
            """
            Compute the inner products using firedrake. Note this
            will be subtracted later, hence appears off by a sign.

            .. math::

                \langle
                    n(x) \cdot \nabla(
                        \int_\Gamma(
                            u(y) \partial_n H_0^{(1)}(\kappa |x - y|)
                        )d\gamma(y)
                    ), v
                \rangle_\Sigma
                - \langle
                    i \kappa \cdot
                    \int_\Gamma(
                        u(y) \partial_n H_0^{(1)}(\kappa |x - y|)
                    )d\gamma(y), v
                \rangle_\Sigma
            """
            self.pyt_result = fd.assemble(
                fd.inner(fd.inner(self.grad_potential_int, self.n),
                         self.v) * fd.ds(outer_bdy_id)
                - fd.inner(self.potential_int, self.v) * fd.ds(outer_bdy_id)
            )

            # y <- Ax - evaluated potential
            self.A.mult(x, y)
            with self.pyt_result.dat.vec_ro as ep:
                y.axpy(-1, ep)

    # {{{ Compute normal helmholtz operator
    firedrake_op.push()

    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)

    """
    .. math::

        \langle
            \nabla u, \nabla v
        \rangle
        - \kappa^2 \cdot \langle
            u, v
        \rangle
        - i \kappa \langle
            u, v
        \rangle_\Sigma
    """
    a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx \
        - kappa**2 * fd.inner(u, v) * fd.dx \
        - 1j * kappa * fd.inner(u, v) * fd.ds(outer_bdy_id)

    # get the concrete matrix from a general bilinear form
    A = fd.assemble(a).M.handle
    # }}}

    # {{{ Setup Python matrix
    B = PETSc.Mat().create()

    # build matrix context
    Bctx = MatrixFreeB(A, pyt_grad_op, pyt_op, queue, kappa)

    # set up B as same size as A
    B.setSizes(*A.getSizes())

    B.setType(B.Type.PYTHON)
    B.setPythonContext(Bctx)
    B.setUp()
    # }}}

    # {{{ Create rhs
    rhs_comp.push()

    # get true solution
    true_sol = fd.Function(V, name="True Solution").interpolate(true_sol_expr)
    true_sol_grad = fd.Function(Vdim).interpolate(fd.grad(true_sol_expr))

    # Remember f is \partial_n(true_sol)|_\Gamma
    # so we just need to compute \int_\Gamma\partial_n(true_sol) H(x-y)
    from pytential import sym

    sigma = sym.make_sym_vector("sigma", ambient_dim)
    """
    ..math:

    x \in \Sigma

    grad_op(x) =
        \nabla(
            \int_\Gamma(
                f(y) H_0^{(1)}(\kappa |x - y|)
            )d\gamma(y)
        )
    """
    grad_op = pyt_inner_normal_sign * \
        sym.grad(2, sym.S(HelmholtzKernel(dim=ambient_dim),
                 sym.n_dot(sigma),
                 k=sym.var("k"), qbx_forced_limit=None))
    """
    ..math:

    x \in \Sigma

    op(x) =
        i \kappa\cdot
        \int_\Gamma(
            f(y) H_0^{(1)}(\kappa |x - y|)
        )d\gamma(y)
        )
    """
    op = 1j * sym.var("k") * pyt_inner_normal_sign * \
        sym.S(HelmholtzKernel(dim=ambient_dim),
                              sym.n_dot(sigma),
                              k=sym.var("k"),
                                   qbx_forced_limit=None)

    rhs_grad_op = fd_bind(function_converter, grad_op,
                          source=(Vdim_dg, inner_bdy_id),
                          target=(Vdim, outer_bdy_id))
    rhs_op = fd_bind(function_converter, op, source=(Vdim_dg, inner_bdy_id),
                     target=(V, outer_bdy_id))

    true_sol_grad_dg = fd.project(true_sol_grad, Vdim_dg,
                                  use_slate_for_inverse=False)
    f_grad_convoluted = rhs_grad_op(queue, sigma=true_sol_grad_dg, k=kappa)
    f_convoluted = rhs_op(queue, sigma=true_sol_grad_dg, k=kappa)
    """
        \langle
            f, v
        \rangle_\Gamma
        - \langle
            n(x) \cdot \nabla(
                \int_\Gamma(
                    f(y) H_0^{(1)}(\kappa |x - y|)
                )d\gamma(y)
            ), v
        \rangle_\Sigma
        + \langle
            i \kappa \cdot \int_\Gamma(
                f(y) H_0^{(1)}(\kappa |x - y|)
            )d\gamma(y), v
        \rangle_\Sigma
    """
    rhs_form = fd.inner(fd.inner(fd.grad(true_sol), fd.FacetNormal(m)),
                        v) * fd.ds(inner_bdy_id) \
        - fd.inner(fd.inner(f_grad_convoluted, fd.FacetNormal(m)),
                   v) * fd.ds(outer_bdy_id) \
        + fd.inner(f_convoluted, v) * fd.ds(outer_bdy_id)

    rhs = fd.assemble(rhs_form)
    rhs_comp.pop()

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
    firedrake_op.pop()

    return solution
