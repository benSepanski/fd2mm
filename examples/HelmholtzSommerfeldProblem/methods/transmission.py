from firedrake import FunctionSpace, Function, TrialFunction, TestFunction, \
    FacetNormal, inner, dot, grad, dx, ds, solve


def transmission(mesh, scatterer_bdy_id, outer_bdy_id, wave_number,
                 fspace=None, true_sol_grad=None):

    u = TrialFunction(fspace)
    v = TestFunction(fspace)
    a = inner(grad(u), grad(v)) * dx - wave_number ** 2 * inner(u, v) * dx \
        - 1j * wave_number * inner(u, v) * ds(outer_bdy_id)

    n = FacetNormal(mesh)
    L = inner(inner(true_sol_grad, n), v) * ds(scatterer_bdy_id)

    solution = Function(fspace)
    solve(a == L, solution)

    return solution
