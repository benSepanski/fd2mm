import firedrake.variational_solver as vs
from firedrake import FunctionSpace, Function, TrialFunction, TestFunction, \
    FacetNormal, inner, dot, grad, dx, ds, Constant, \
    assemble
from firedrake.exceptions import ConvergenceError
from .preconditioners.two_D_helmholtz import AMGTransmissionPreconditioner


def transmission(mesh, scatterer_bdy_id, outer_bdy_id, wave_number,
                 options_prefix=None, solver_parameters=None,
                 fspace=None, true_sol_grad=None,
                 ):
    r"""
        preconditioner_gamma and preconditioner_lambda are used to precondition
        with the following equation:

        \Delta u - \kappa^2 \gamma u = 0
        (\partial_n - i\kappa\beta) u |_\Sigma = 0
    """

    u = TrialFunction(fspace)
    v = TestFunction(fspace)
    a = inner(grad(u), grad(v)) * dx - Constant(wave_number**2) * inner(u, v) * dx \
        - Constant(1j * wave_number) * inner(u, v) * ds(outer_bdy_id)

    n = FacetNormal(mesh)
    L = inner(inner(true_sol_grad, n), v) * ds(scatterer_bdy_id)

    solution = Function(fspace)

    #       {{{ Used for preconditioning
    if 'gamma' in solver_parameters or 'beta' in solver_parameters:
        solver_params = dict(solver_parameters)
        gamma = complex(solver_parameters.pop('gamma', 1.0))

        import cmath
        beta = complex(solver_parameters.pop('beta', cmath.sqrt(gamma)))

        aP = inner(grad(u), grad(v)) * dx \
            - Constant(wave_number**2 * gamma) * inner(u, v) * dx \
            - Constant(1j * wave_number * beta) * inner(u, v) * ds(outer_bdy_id)
    else:
        aP = None
        solver_params = solver_parameters
    #       }}}

    # prepare to set up pyamg preconditioner if using it
    using_pyamg = solver_params['pc_type'] == 'pyamg'
    if using_pyamg:
        pyamg_tol = solver_parameters.get('pyamg_tol', None)
        if pyamg_tol is not None:
            pyamg_tol = float(pyamg_tol)
        pyamg_maxiter = solver_params.get('pyamg_maxiter', None)
        if pyamg_maxiter is not None:
            pyamg_maxiter = int(pyamg_maxiter)
        del solver_params['pc_type']

    # Create a solver and return the KSP object with the solution so that can get
    # PETSc information
    # Create problem
    problem = vs.LinearVariationalProblem(a, L, solution, aP=aP)

    # Create solver and call solve
    solver = vs.LinearVariationalSolver(problem, solver_parameters=solver_params,
                                        options_prefix=options_prefix)
    # prepare to set up pyamg preconditioner if using it
    if using_pyamg:
        A = assemble(a).M.handle
        pc = solver.snes.getKSP().pc
        pc.setType(pc.Type.PYTHON)
        pc.setPythonContext(AMGTransmissionPreconditioner(wave_number,
                                                          fspace,
                                                          A,
                                                          tol=pyamg_tol,
                                                          maxiter=pyamg_maxiter,
                                                          use_plane_waves=True))

    # If using pyamg as preconditioner, use it!
    try:
        solver.solve()
    except ConvergenceError:
        pass

    return solver.snes, solution
