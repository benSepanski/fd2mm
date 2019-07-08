from firedrake import Function, FacetNormal, TestFunction, assemble, inner, ds, \
    TrialFunction, grad, dx, Constant
from firedrake.petsc import PETSc
from firedrake_to_pytential.op import fd_bind
from sumpy.kernel import HelmholtzKernel


def nonlocal_integral_eq(mesh, scatterer_bdy_id, outer_bdy_id, wave_number,
                         fspace=None, vfspace=None,
                         true_sol=None, true_sol_grad=None,
                         cl_ctx=None, queue=None, function_converter=None,
                         eta=None):
    """
        see run_method for descriptions of unlisted args

        args:

        :arg cl_ctx: A pyopencl computing context
        :arg queue: A command queue for the computing context
        :arg function_converter: A function converter from firedrake to pytential
        :arg eta: Used in Robin condition, defaults tow ave number
    """
    if eta is None:
        eta = wave_number

    # away from the excluded region, but firedrake and meshmode point
    # into
    pyt_inner_normal_sign = -1

    ambient_dim = mesh.geometric_dimension()

    # {{{ Create operator
    from pytential import sym

    r"""
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
        ambient_dim, sym.D(HelmholtzKernel(ambient_dim),
                           sym.var("u"), k=sym.var("k"),
                           qbx_forced_limit=None))

    r"""
    ..math:

    x \in \Sigma

    op(x) =
        i \eta \cdot
        \int_\Gamma(
            u(y) \partial_n H_0^{(1)}(\kappa |x - y|)
        )d\gamma(y)
    """
    op = pyt_inner_normal_sign * 1j * sym.var("eta") * (
        sym.D(HelmholtzKernel(ambient_dim),
                sym.var("u"), k=sym.var("k"),
                qbx_forced_limit=None)
        )

    pyt_grad_op = fd_bind(function_converter, grad_op,
                          source=(fspace, scatterer_bdy_id),
                          target=(vfspace, outer_bdy_id))
    pyt_op = fd_bind(function_converter, op, source=(fspace, scatterer_bdy_id),
                         target=(fspace, outer_bdy_id))
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
            self.x_fntn = Function(fspace)

            self.potential_int = Function(fspace)
            self.potential_int.dat.data[:] = 0.0
            self.grad_potential_int = Function(vfspace)
            self.grad_potential_int.dat.data[:] = 0.0
            self.pyt_result = Function(fspace)

            self.n = FacetNormal(mesh)
            self.v = TestFunction(fspace)
            # }}}

        def mult(self, mat, x, y):
            # Perform pytential operation
            self.x_fntn.dat.data[:] = x[:]

            self.pyt_op(self.queue, result_function=self.potential_int,
                        u=self.x_fntn, k=self.k, eta=eta)
            self.pyt_grad_op(self.queue, result_function=self.grad_potential_int,
                             u=self.x_fntn, k=self.k)

            # Integrate the potential
            r"""
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
                    i \eta \cdot
                    \int_\Gamma(
                        u(y) \partial_n H_0^{(1)}(\kappa |x - y|)
                    )d\gamma(y), v
                \rangle_\Sigma
            """
            self.pyt_result = assemble(
                inner(inner(self.grad_potential_int, self.n),
                      self.v) * ds(outer_bdy_id)
                - inner(self.potential_int, self.v) * ds(outer_bdy_id)
            )

            # y <- Ax - evaluated potential
            self.A.mult(x, y)
            with self.pyt_result.dat.vec_ro as ep:
                y.axpy(-1, ep)

    # {{{ Compute normal helmholtz operator
    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    r"""
    .. math::

        \langle
            \nabla u, \nabla v
        \rangle
        - \kappa^2 \cdot \langle
            u, v
        \rangle
        - i \eta \langle
            u, v
        \rangle_\Sigma
    """
    a = inner(grad(u), grad(v)) * dx \
        - wave_number**2 * inner(u, v) * dx \
        - Constant(1j * eta) * inner(u, v) * ds(outer_bdy_id)

    # get the concrete matrix from a general bilinear form
    A = assemble(a).M.handle
    # }}}

    # {{{ Setup Python matrix
    B = PETSc.Mat().create()

    # build matrix context
    Bctx = MatrixFreeB(A, pyt_grad_op, pyt_op, queue, wave_number)

    # set up B as same size as A
    B.setSizes(*A.getSizes())

    B.setType(B.Type.PYTHON)
    B.setPythonContext(Bctx)
    B.setUp()
    # }}}

    # {{{ Create rhs

    # Remember f is \partial_n(true_sol)|_\Gamma
    # so we just need to compute \int_\Gamma\partial_n(true_sol) H(x-y)
    from pytential import sym

    sigma = sym.make_sym_vector("sigma", ambient_dim)
    r"""
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
        sym.grad(ambient_dim, sym.S(HelmholtzKernel(ambient_dim),
                                    sym.n_dot(sigma),
                                    k=sym.var("k"), qbx_forced_limit=None))
    r"""
    ..math:

    x \in \Sigma

    op(x) =
        i \eta\cdot
        \int_\Gamma(
            f(y) H_0^{(1)}(\kappa |x - y|)
        )d\gamma(y)
        )
    """
    op = 1j * sym.var("eta") * pyt_inner_normal_sign * \
        sym.S(HelmholtzKernel(dim=ambient_dim),
                              sym.n_dot(sigma),
                              k=sym.var("k"),
                                   qbx_forced_limit=None)

    rhs_grad_op = fd_bind(function_converter, grad_op,
                          source=(vfspace, scatterer_bdy_id),
                          target=(vfspace, outer_bdy_id))
    rhs_op = fd_bind(function_converter, op, source=(vfspace, scatterer_bdy_id),
                     target=(fspace, outer_bdy_id))

    f_grad_convoluted = rhs_grad_op(queue, sigma=true_sol_grad, k=wave_number)
    f_convoluted = rhs_op(queue, sigma=true_sol_grad, k=wave_number, eta=eta)
    r"""
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
            i \eta \cdot \int_\Gamma(
                f(y) H_0^{(1)}(\kappa |x - y|)
            )d\gamma(y), v
        \rangle_\Sigma
    """
    rhs_form = inner(inner(grad(true_sol), FacetNormal(mesh)),
                     v) * ds(scatterer_bdy_id) \
        - inner(inner(f_grad_convoluted, FacetNormal(mesh)),
                v) * ds(outer_bdy_id) \
        + inner(f_convoluted, v) * ds(outer_bdy_id)

    rhs = assemble(rhs_form)

    # {{{ set up a solver:
    solution = Function(fspace, name="Computed Solution")

    ksp = PETSc.KSP().create()
    # precondition with A
    ksp.setOperators(B, A)

    ksp.setFromOptions()
    with rhs.dat.vec_ro as b:
        with solution.dat.vec as x:
            ksp.solve(b, x)
    # }}}

    return solution
