from firedrake import Constant, SpatialCoordinate, as_tensor, \
    Function, TrialFunction, TestFunction, \
    inner, grad, solve, dx, ds, DirichletBC, dot, FacetNormal


def pml(mesh, scatterer_bdy_id, outer_bdy_id, wave_number,
        inner_region=None, pml_x_region=None, pml_y_region=None, pml_xy_region=None,
        fspace=None, tfspace=None, true_sol_grad=None,
        pml_type=None, delta=None, quad_const=None, speed=None,
        pml_x_min=None, pml_y_min=None, pml_x_max=None, pml_y_max=None,
        options_prefix=None):
    """
        For unlisted arg descriptions, see run_method

        :arg inner_region: boundary id of non-pml region
        :arg pml_x_region: boundary id of region where pml is only required
                           in the x direction
        :arg pml_y_region: boundary id of region where pml is only required
                           in the y direction
        :arg pml_xy_region: boundary id of region where pml is required
                           in the x and y direction
        :arg pml_type: Type of pml function, either 'quadratic' or 'bdy_integral'
        :arg delta: For :arg:`pml_type` of 'bdy_integral', added to denominator
                    to prevent 1 / 0 at edge of boundary
        :arg quad_const: For :arg:`pml_type` of 'quadratic', a scaling constant
        :arg speed: Speed of sound
        :arg pml_x_min: Left edge where to begin pml
        :arg pml_y_min: Bottom edge where to begin pml
        :arg pml_x_max: Right edge where to stop pml
        :arg pml_y_max: Top edge where to stop pml
    """
    # Handle defauls
    if pml_type is None:
        pml_type = 'bdy_integral'
    if delta is None:
        delta = 1e-3
    if quad_const is None:
        quad_const = 1.0
    if speed is None:
        speed = 340.0

    if options_prefix is None:
        options_prefix = ''

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
            - Constant(k**2) * inner(p, q)
         ) * dx(inner_region) + \
        (inner(dot(grad(p), kappa_xy), grad(q))
            - Constant(k**2) * gamma_x * gamma_y * inner(p, q)
         ) * dx(pml_xy_region) + \
        (inner(dot(grad(p), kappa_x), grad(q))
            - Constant(k**2) * gamma_x * inner(p, q)
         ) * dx(pml_x_region) + \
        (inner(dot(grad(p), kappa_y), grad(q))
            - Constant(k**2) * gamma_y * inner(p, q)
         ) * dx(pml_y_region)

    n = FacetNormal(mesh)
    L = inner(dot(true_sol_grad, n), q) * ds(scatterer_bdy_id)

    bc = DirichletBC(fspace, Constant(0), outer_bdy_id)

    solution = Function(fspace)
    solve(a == L, solution, bcs=[bc], options_prefix=options_prefix)

    return solution
