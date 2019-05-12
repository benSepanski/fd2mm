from firedrake import *
import numpy as np
from pml_functions import *
from math import factorial, pi

mesh = Mesh("domain.msh")

V = FunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)

zero = np.zeros(1, dtype=np.complex128)

a_prime = 3 # magnitude of right/left side of outer rectangle
b_prime = 3 # magnitude of top/bottom side of outer rectangle

k=750/340
c=340
quad_const = Constant(40 * c)

# g defined so that solution on bermudez_paper_domain.msh is
#  p(x,y) = j/4 * H_0^1(k \sqrt((x-0.5)^2 + y^2))
# where H_0^1 is a Hankel function of the first kind
imag_unit = Constant((np.zeros(1, dtype=np.complex128) + 1j)[0])
h_0 = imag_unit / 4 * hankel_function(k * sqrt((x - 0.5)**2 + y**2))
V_squared = VectorFunctionSpace(V.mesh(), V.ufl_element())
grad_h = Function(V_squared).interpolate(grad(h_0))

u = TrialFunction(V)
v = conj(TestFunction(V))

inner_bdy = 5
outer_bdy = 6

a = (dot(grad(u), conj(grad(v))) - k**2 * u * v) * dx - (imag_unit * k * u * v ) * ds(outer_bdy)
n = FacetNormal(V.mesh())
L = dot(grad_h, conj(n)) * v * ds(inner_bdy)

p = Function(V)

amg_params = {
    "ksp_type": "gmres",
    "ksp_monitor": True,
    "pc_type": "gamg"
}

direct_params = {
    "ksp_type": "preonly",
    "pc_type": "lu"
}

# solve(a == L, p, solver_parameters = amg_params)
solve(a == L, p, solver_parameters = direct_params)

inner_region = 4

f = Function(V).interpolate(h_0)
diff = sqrt(assemble(inner(f - p, f - p)*dx(inner_region)))
sol_norm = sqrt(assemble(inner(f, f)*dx(inner_region)))
print("L^2 error in non-PML region:", diff)
print("L^2 relative error in non-PML region (%):", diff / sol_norm * 100)



