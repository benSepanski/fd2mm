from firedrake import FunctionSpace, Function, TrialFunction, TestFunction, \
    FacetNormal, inner, dot, grad, dx, ds, solve


def transmission(wave_number, **kwargs):
    inner_bdy_id = kwargs['scatterer_bdy_id']
    outer_bdy_id = kwargs['outer_bdy_id']

    fspace = kwargs['fspace']
    mesh = kwargs['mesh']
    true_sol_grad = kwargs['true_sol_grad']

    u = TrialFunction(fspace)
    v = TestFunction(fspace)
    a = inner(grad(u), grad(v)) * dx - wave_number ** 2 * inner(u, v) * dx \
        - 1j * wave_number * inner(u, v) * ds(outer_bdy_id)

    n = FacetNormal(mesh)
    L = inner(dot(true_sol_grad, n), v) * ds(inner_bdy_id)

    solution = Function(fspace)
    solve(a == L, solution)

    return solution
