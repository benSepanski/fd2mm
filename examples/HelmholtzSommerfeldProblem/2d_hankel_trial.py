import os
import pyopencl as cl

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

# For WSL, all firedrake must be imported after pyopencl
import firedrake as fd
from hankel_function import hankel_function
from run_trial import run_trial

# Trial settings
mesh_file_dir = "circle_in_square/"  # NEED a forward slash at end
kappa_list = [3.0]
degree_list = [1]
method_list = ['nonlocal_integral_eq']
method_to_kwargs = {
    'transmission': {},
    'pml': {
        'pml_type': 'bdy_integral',
        },
    'nonlocal_integral_eq': {
        'cl_ctx': cl_ctx,
        'queue': queue,
        'with_refinement': True}
    }


# Plot computed and true solutions?
visualize = False
if visualize:
    import matplotlib.pyplot as plt


# Hankel approximation cutoff
hankel_cutoff = 25

inner_bdy_id = 1
outer_bdy_id = 2
inner_region = 3
pml_x_region = 4
pml_y_region = 5
pml_xy_region = 6

pml_x_min = 2
pml_x_max = 3
pml_y_min = 2
pml_y_max = 3

# Set kwargs that don't expect user to change
# (NOTE: some of these are for just pml, but we don't
#  expect the user to want to change them
global_kwargs = {'scatterer_bdy_id': inner_bdy_id,
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

# Go ahead and make the file directory accurate
mesh_file_dir = 'meshes/' + mesh_file_dir

# Ready kwargs by adding global kwargs to them
for mkey in method_to_kwargs:
    for gkey in global_kwargs:
        method_to_kwargs[mkey][gkey] = global_kwargs[gkey]

print("Reading Meshes...")
meshes = []
for filename in os.listdir(mesh_file_dir):
    if filename.endswith('.msh'):
        meshes.append(fd.Mesh(mesh_file_dir + filename))
meshes.sort(key=lambda x: x.coordinates.dat.data.shape[0])
print("Meshes Read in.")


def get_key(setup_info):
    key = (setup_info['mesh'],
           setup_info['kappa'],
           setup_info['degree_list'],
           setup_info['method'])
    return key


def relative_error(true_sol, comp_sol):
    true_sol_norm = fd.sqrt(fd.assemble(
        fd.inner(true_sol, true_sol) * fd.dx(inner_region)
        ))
    l2_err = fd.sqrt(fd.assemble(
        fd.inner(true_sol - comp_sol, true_sol - comp_sol) * fd.dx(inner_region)
        ))
    return l2_err / true_sol_norm


for mesh in meshes:
    x, y = fd.SpatialCoordinate(mesh)

    for degree in degree_list:

        for kappa in kappa_list:
            true_sol_expr = fd.Constant(1j / 4) * \
                hankel_function(kappa * fd.sqrt((x - 0.5)**2 + y**2),
                                n=hankel_cutoff)

            trial = {'mesh': mesh,
                     'degree': degree,
                     'true_sol_expr': true_sol_expr}

            for method in method_list:
                kwargs = method_to_kwargs[method]

                true_sol, comp_sol = run_trial(trial, method, kappa, **kwargs,
                                               comp_sol_name=
                                               method + " Computed Solution")
                # Plot if visualizing
                if visualize:
                    fd.plot(comp_sol)
                    fd.plot(true_sol)
                    plt.show()

                rel_err = relative_error(true_sol, comp_sol)
                print("ndofs:", true_sol.dat.data.shape[0])
                print("kappa:", kappa)
                print("method:", method)
                print('degree:', degree)
                print("L^2 Relative Err: ", rel_err)
                print()
