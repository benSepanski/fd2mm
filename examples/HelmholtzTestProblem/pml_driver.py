from firedrake import *
import numpy as np
from pml_functions import *
from math import factorial, pi

def pml(sigma_function, V, kappa, c, true_sol_expr,
        outer_bdy, inner_bdy, inner_region,
        pml_x_region, pml_y_region,
        pml_xy_region, **kwargs):
    """
        sigma_function: type of sigma function to use

        kwarg quad_min: max quad_const to check in [quad_min, quad_max]
        kwarg quad_max: max quad_const to check in [quad_min, quad_max]
        kwarg quad_step: step by which to check quad_const
        kwarg delta: added to denominator of bdy_int sigma function to
                     help avoid discretization errors
        kwarg solve_params: solver params
    """
    k = kappa
    family = "CG"
    degree = V.ufl_element().degree()
    mesh = V.mesh()

    x, y = SpatialCoordinate(mesh)

    a = 2 # magnitude of right/left side of inner rectangle
    b = 2 # magnitude of top/bottom side of inner rectangle

    zero = np.zeros(1, dtype=np.complex128)

    a_prime = 3 # magnitude of right/left side of outer rectangle
    b_prime = 3 # magnitude of top/bottom side of outer rectangle

    quad_max = kwargs.get("quad_max", 1)
    quad_min = kwargs.get("quad_min", 1)
    quad_step = kwargs.get("quad_step", 1)

    scale = quad_min
    results = []
    while scale <= quad_max:
        if sigma_function == "quadratic":
            quad_const = Constant(scale * c)
            sigma_x = quad_const * (abs(x) - a)**2
            sigma_y = quad_const * (abs(y) - b)**2
        elif sigma_function == "bdy_int":
            delta = Constant(kwargs.get("delta", 0.1))
            sigma_x = c / (delta + a_prime - abs(x))
            sigma_y = c / (delta + b_prime - abs(y))
        else:
            print("invalid sigma function, please use \"quadratic\" or \"bdy_int\"")
            return [0, 0]

        h_0 = Function(V).interpolate(true_sol_expr)

        result = pml_solver(h_0, k, c, V, sigma_x, sigma_y,
                            outer_bdy=outer_bdy,
                            inner_bdy=inner_bdy,
                            inner_region=inner_region,
                            pml_x_region=pml_x_region,
                            pml_y_region=pml_y_region,
                            pml_xy_region=pml_xy_region,
                            solver_params=kwargs.get('solver_params', None),
                            print_error=False)
        results.append(result)

        if sigma_function != "quadratic":
            break
        scale += quad_step

    if len(results) == 1:
        result = results[0]
    else:
        errs = [assemble(inner(r - h_0, r - h_0) * dx(inner_region)) for r in results]
        ind = 0
        for i, e in enumerate(errs):
            if e < errs[ind]:
                ind = i
        result = results[ind]

    return result
