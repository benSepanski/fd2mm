import firedrake.variational_solver as vs
from firedrake import FunctionSpace, Function, TrialFunction, TestFunction, \
    FacetNormal, inner, dot, grad, dx, ds, Constant


def transmission(mesh, scatterer_bdy_id, outer_bdy_id, wave_number,
                 options_prefix=None, solver_parameters=None,
                 fspace=None, true_sol_grad=None):

    u = TrialFunction(fspace)
    v = TestFunction(fspace)
    a = inner(grad(u), grad(v)) * dx - Constant(wave_number**2) * inner(u, v) * dx \
        - Constant(1j * wave_number) * inner(u, v) * ds(outer_bdy_id)

    n = FacetNormal(mesh)
    L = inner(inner(true_sol_grad, n), v) * ds(scatterer_bdy_id)

    solution = Function(fspace)

    # Create a solver and return the KSP object with the solution so that can get
    # PETSc information
    # Create problem
    problem = vs.LinearVariationalProblem(a, L, solution, (), None)
    # Create solver and call solve
    solver = vs.LinearVariationalSolver(problem, solver_parameters=solver_parameters,
                                        options_prefix=options_prefix)
    solver.solve()

    return solver.snes.ksp, solution
