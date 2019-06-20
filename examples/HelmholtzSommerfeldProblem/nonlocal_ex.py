import pyopencl as cl

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

import firedrake as fd
from hankel_function import hankel_function
from run_trial import run_trial


mesh = fd.Mesh('domain.msh')
pml_x_region = 1
pml_y_region = 2
pml_xy_region = 3
inner_region = 4
outer_bdy_id = 6
inner_bdy_id = 5

x, y = fd.SpatialCoordinate(mesh)

kappa = 2.7
true_sol_expr = fd.Constant(1j / 4) * \
    hankel_function(kappa * fd.sqrt((x - 0.5)**2 + y**2), n=25)

trial = {'mesh': mesh,
         'degree': 2,
         'true_sol_expr': true_sol_expr}
kwargs = {'scatterer_bdy_id': inner_bdy_id,
          'outer_bdy_id': outer_bdy_id,
          'cl_ctx': cl_ctx,
          'queue': queue,
          'with_refinement': False}

true_sol, comp_sol = run_trial(trial, 'nonlocal_integral_eq', kappa, **kwargs)

true_sol_norm = fd.sqrt(fd.assemble(
    fd.inner(true_sol, true_sol) * fd.dx
    ))
l2_err = fd.sqrt(fd.assemble(
    fd.inner(true_sol - comp_sol, true_sol - comp_sol) * fd.dx
    ))
rel_err = l2_err / true_sol_norm

print("L2 relative error=", rel_err)
