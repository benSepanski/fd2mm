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

pml_x_min = 2
pml_x_max = 3
pml_y_min = 2
pml_y_max = 3

x, y = fd.SpatialCoordinate(mesh)

kappa = 2.7
true_sol_expr = fd.Constant(1j / 4) * \
    hankel_function(kappa * fd.sqrt((x - 0.5)**2 + y**2), n=25)

trial = {'mesh': mesh,
         'degree': 2,
         'true_sol_expr': true_sol_expr}
kwargs = {'scatterer_bdy_id': inner_bdy_id,
          'outer_bdy_id': outer_bdy_id,
          'inner_region': inner_region,
          'pml_x_region': pml_x_region,
          'pml_y_region': pml_y_region,
          'pml_xy_region': pml_xy_region,
          'pml_x_min': pml_x_min,
          'pml_x_max': pml_x_max,
          'pml_y_min': pml_y_min,
          'pml_y_max': pml_y_max,
          }

true_sol, comp_sol = run_trial(trial, 'pml', kappa, **kwargs)

true_sol_norm = fd.sqrt(fd.assemble(
    fd.inner(true_sol, true_sol) * fd.dx
    ))
l2_err = fd.sqrt(fd.assemble(
    fd.inner(true_sol - comp_sol, true_sol - comp_sol) * fd.dx
    ))
rel_err = l2_err / true_sol_norm

print("L2 relative error=", rel_err)
