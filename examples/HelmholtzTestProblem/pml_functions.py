from firedrake import *
import numpy as np
from math import factorial, pi

# uses pml to solve helmholtz equation

def pml_solver(f, k, c, V, sigma_x, sigma_y, outer_bdy=1, inner_bdy=2, inner_region=3, pml_x_region=4,pml_y_region=5,pml_xy_region=6, print_error=False, **kwargs):
    """
    f : forcing function
    k : diffusion coefficient
    c : speed of wave
    V : function space
    sigma_x : variable diffusion coeffient for x direction
    sigma_y : variable diffusion coefficient for y direction
    outer_bdy: subdomain id of outer boundary

    kwargs:
    solver_params
    """
    omega = k * c
    
    gamma_x = (1 + 1j / omega * sigma_x)
    gamma_y = (1 + 1j / omega * sigma_y)

    
    kappa_xy = as_tensor([[gamma_y / gamma_x, 0], [0, gamma_x / gamma_y]])
    kappa_x = as_tensor([[1 / gamma_x, 0], [0, gamma_x]])
    kappa_y = as_tensor([[gamma_y, 0], [0, 1 / gamma_y]])
       
    W = TensorFunctionSpace(V.mesh(), V.ufl_element())
    kappa_x = Function(W).interpolate(kappa_x)
    kappa_y = Function(W).interpolate(kappa_y)
    kappa_xy = Function(W).interpolate(kappa_xy)

    p = TrialFunction(V) # trial function
    q = TestFunction(V) # test function

    a = (inner(grad(p), grad(q)) - k ** 2 * inner(p, q)) * dx(inner_region) + \
        (inner(dot(grad(p), kappa_xy), grad(q)) - k**2 * gamma_x * gamma_y * inner(p, q)
        ) * dx(pml_xy_region) + \
        (inner(dot(grad(p), kappa_x), grad(q)) - k**2 * gamma_x * inner(p, q)
        ) * dx(pml_x_region) + \
        (inner(dot(grad(p), kappa_y), grad(q)) - k**2 * gamma_y * inner(p, q)
        ) * dx(pml_y_region)
    
    n = FacetNormal(V.mesh())
    V_squared = VectorFunctionSpace(V.mesh(), V.ufl_element())
    grad_f = Function(V_squared).interpolate(grad(f))
    L= inner(dot(grad_f, conj(n)), q) * ds(inner_bdy)

    bc = DirichletBC(V, Constant(0), outer_bdy)
    p = Function(V)

    params = kwargs.get('solver_params', None)
    if params is None:
        params = {"ksp_monitor": True,
                  "ksp_gmres_restart": 200,
                  "pc_type": "lu",
                  "ksp_max_it": 500}
    
    solve(a == L, p, bcs=[bc], solver_parameters=params)
   
    if print_error:
        f = Function(V).interpolate(f)
        diff = sqrt(assemble(inner(f - p, f - p)*dx(inner_region)))
        sol_norm = sqrt(assemble(inner(f, f)*dx(inner_region)))
        print("L^2 error in non-PML region:", diff)
        print("L^2 relative error in non-PML region (%):", diff / sol_norm * 100)
        print("d.o.f. :", V.dim())
    return p



def hankel_function(e, n=None):
    if n is None:
        n = 97
    j_0 = 0
    for i in range(n):
        j_0 += (-1)**i * (1 / 4 * e**2)**i / factorial(i)**2

    g = 0.57721566490153286
    y_0 = (ln(e / 2) + g) * j_0
    h_n = 0
    for i in range(n):
        h_n += 1 / (i + 1)
        y_0 += (-1)**(i) * h_n * (e**2 / 4)**(i+1) / (factorial(i+1))**2
    y_0 *= 2 / pi
   
    imag_unit = Constant((np.zeros(1,dtype=np.complex128) + 1j)[0])
    h_0 = j_0 + imag_unit * y_0
    return h_0

def eval_hankel_function(x, n=97):
    j_0 = 0
    for i in range(n):
        j_0 += (-1)**i * (1 / 4 * e**2)**i / factorial(i)**2

    g = 0.57721566490153286
    y_0 = (ln(e / 2) + g) * j_0
    h_n = 0
    for i in range(n):
        h_n += 1 / (i + 1)
        y_0 += (-1)**(i) * h_n * (e**2 / 4)**(i+1) / (factorial(i+1))**2
    y_0 *= 2 / pi
    
    imag_unit = (np.zeros(1,dtype=np.complex128) + 1j)[0]
    h_0 = j_0 + imag_unit * y_0
    return h_0

