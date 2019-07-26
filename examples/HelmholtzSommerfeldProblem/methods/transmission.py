import firedrake.variational_solver as vs
from firedrake import FunctionSpace, Function, TrialFunction, TestFunction, \
    FacetNormal, inner, dot, grad, dx, ds, Constant


def transmission(mesh, scatterer_bdy_id, outer_bdy_id, wave_number,
                 options_prefix=None, solver_parameters=None,
                 fspace=None, true_sol_grad=None):
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

    # Create a solver and return the KSP object with the solution so that can get
    # PETSc information
    # Create problem
    problem = vs.LinearVariationalProblem(a, L, solution, aP=aP)
    # Create solver and call solve
    solver = vs.LinearVariationalSolver(problem, solver_parameters=solver_params,
                                        options_prefix=options_prefix)
    solver.solve()

    return solver.snes, solution
