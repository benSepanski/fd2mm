from firedrake import Constant, SpatialCoordinate, as_tensor, \
    Function, TrialFunction, TestFunction, \
    inner, grad, solve, dx, ds, DirichletBC, dot, FacetNormal


def pml(wave_number, **kwargs):
    pml_type = kwargs['pml_type']
    delta = kwargs['delta']
    quad_const = kwargs['quad_const']

    mesh = kwargs['mesh']
    fspace = kwargs['fspace']
    tfspace = kwargs['tfspace']
    true_sol_grad = kwargs['true_sol_grad']

    speed = kwargs['speed']

    pml_x_min = kwargs['pml_x_min']
    pml_y_min = kwargs['pml_y_min']
    pml_x_max = kwargs['pml_x_max']
    pml_y_max = kwargs['pml_y_max']

    inner_bdy = kwargs['scatterer_bdy_id']
    outer_bdy = kwargs['outer_bdy_id']
    inner_region = kwargs['inner_region']
    pml_x_region = kwargs['pml_x_region']
    pml_y_region = kwargs['pml_y_region']
    pml_xy_region = kwargs['pml_xy_region']

    pml_types = ['bdy_integral', 'quadratic']
    if pml_type not in pml_types:
        raise ValueError("PML type of %s is not one of %s" % (pml_type, pml_types))

    if mesh.geometric_dimension() != 2:
        raise ValueError("PML only implemented in 2D")

    x, y = SpatialCoordinate(mesh)
    # {{{ create sigma functions for PML
    if pml_type == 'bdy_integral':
        sigma_x = speed / (delta + pml_x_max - abs(x))
        sigma_y = speed / (delta + pml_y_max - abs(y))
    elif pml_type == 'quadratic':
        sigma_x = quad_const * (abs(x) - pml_x_min) ** 2
        sigma_y = quad_const * (abs(y) - pml_y_min) ** 2

    r"""
        Here \kappa is the wave number and c is the speed

        ..math::

        \kappa = \frac{ \omega } { c }
    """
    omega = wave_number * speed

    # {{{ Set up PML functions
    gamma_x = (1 + 1j / omega * sigma_x)
    gamma_y = (1 + 1j / omega * sigma_y)

    kappa_xy = as_tensor([[gamma_y / gamma_x, 0], [0, gamma_x / gamma_y]])
    kappa_x = as_tensor([[1 / gamma_x, 0], [0, gamma_x]])
    kappa_y = as_tensor([[gamma_y, 0], [0, 1 / gamma_y]])

    kappa_x = Function(tfspace).interpolate(kappa_x)
    kappa_y = Function(tfspace).interpolate(kappa_y)
    kappa_xy = Function(tfspace).interpolate(kappa_xy)

    # }}}

    p = TrialFunction(fspace)
    q = TestFunction(fspace)

    k = wave_number  # Just easier to look at
    a = (inner(grad(p), grad(q))
            - k ** 2 * inner(p, q)
         ) * dx(inner_region) + \
        (inner(dot(grad(p), kappa_xy), grad(q))
            - k**2 * gamma_x * gamma_y * inner(p, q)
         ) * dx(pml_xy_region) + \
        (inner(dot(grad(p), kappa_x), grad(q))
            - k**2 * gamma_x * inner(p, q)
         ) * dx(pml_x_region) + \
        (inner(dot(grad(p), kappa_y), grad(q))
            - k**2 * gamma_y * inner(p, q)
         ) * dx(pml_y_region)

    n = FacetNormal(fspace.mesh())
    L = inner(dot(true_sol_grad, n), q) * ds(inner_bdy)

    bc = DirichletBC(fspace, Constant(0), outer_bdy)

    solution = Function(fspace)
    solve(a == L, solution, bcs=[bc])

    return solution
