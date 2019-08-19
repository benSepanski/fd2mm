import firedrake.variational_solver as vs
from firedrake import Constant, SpatialCoordinate, as_tensor, \
    Function, TrialFunction, TestFunction, \
    inner, grad, solve, dx, ds, DirichletBC, dot, FacetNormal, \
    conditional, real


def pml(mesh, scatterer_bdy_id, outer_bdy_id, wave_number,
        options_prefix=None, solver_parameters=None,
        inner_region=None,
        fspace=None, tfspace=None, true_sol_grad=None,
        pml_type=None, delta=None, quad_const=None, speed=None,
        pml_min=None, pml_max=None):
    """
        For unlisted arg descriptions, see run_method

        :arg inner_region: boundary id of non-pml region
        :arg pml_type: Type of pml function, either 'quadratic' or 'bdy_integral'
        :arg delta: For :arg:`pml_type` of 'bdy_integral', added to denominator
                    to prevent 1 / 0 at edge of boundary
        :arg quad_const: For :arg:`pml_type` of 'quadratic', a scaling constant
        :arg speed: Speed of sound
        :arg pml_min: A list, *pml_min[i]* is where to begin pml layer in direction
                      *i*
        :arg pml_max: A list, *pml_max[i]* is where to end pml layer in direction *i*
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

    pml_types = ['bdy_integral', 'quadratic']
    if pml_type not in pml_types:
        raise ValueError("PML type of %s is not one of %s" % (pml_type, pml_types))

    xx = SpatialCoordinate(mesh)
    # {{{ create sigma functions for PML
    sigma = None
    if pml_type == 'bdy_integral':
        sigma = [Constant(speed) / (Constant(delta + extent) - abs(coord)) for
                 extent, coord in zip(pml_max, xx)]
    elif pml_type == 'quadratic':
        sigma = [Constant(quad_const) * (abs(coord) - Constant(min_)) ** 2
                 for min_, coord in zip(pml_min, xx)]

    r"""
        Here \kappa is the wave number and c is the speed

        ..math::

        \kappa = \frac{ \omega } { c }
    """
    omega = wave_number * speed

    # {{{ Set up PML functions
    gamma = [Constant(1.0) + conditional(abs(real(coord)) >= real(min_),
                                         Constant(1j / omega) * sigma_i,
                                         Constant(0.0))
             for min_, coord, sigma_i in zip(pml_min, xx, sigma)]

    kappa = [None] * len(gamma)
    gamma_prod = 1.0
    for i in range(len(gamma)):
        gamma_prod *= gamma[i]
        tensor_i = [Constant(0.0) for _ in range(len(gamma))]
        tensor_i[i] = 1.0
        r"""
            *i*th entry is

            .. math::

            \frac{\prod_{j\neq i} \gamma_j}{ \gamma_i }
        """
        for j in range(len(gamma)):
            if j != i:
                tensor_i[i] *= gamma[j]
            else:
                tensor_i[i] /= gamma[j]
        kappa[i] = tensor_i

    kappa = as_tensor(kappa)

    # }}}

    p = TrialFunction(fspace)
    q = TestFunction(fspace)

    k = wave_number  # Just easier to look at
    a = (inner(dot(grad(p), kappa), grad(q))
            - Constant(k**2) * gamma_prod * inner(p, q)
         ) * dx

    n = FacetNormal(mesh)
    L = inner(dot(true_sol_grad, n), q) * ds(scatterer_bdy_id)

    bc = DirichletBC(fspace, Constant(0), outer_bdy_id)

    solution = Function(fspace)

    #solve(a == L, solution, bcs=[bc], options_prefix=options_prefix)
    # Create a solver and return the KSP object with the solution so that can get
    # PETSc information
    # Create problem
    problem = vs.LinearVariationalProblem(a, L, solution, [bc], None)
    # Create solver and call solve
    solver = vs.LinearVariationalSolver(problem, solver_parameters=solver_parameters,
                                        options_prefix=options_prefix)
    solver.solve()

    return solver.snes, solution
