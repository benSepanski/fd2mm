import os
import csv
import matplotlib.pyplot as plt
import pyopencl as cl

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

# For WSL, all firedrake must be imported after pyopencl
import firedrake as fd
from utils.hankel_function import hankel_function
import utils.norm_functions as norms
from methods.run_method import run_method

# {{{ Trial settings for user to modify

mesh_file_dir = "circle_in_square/"  # NEED a forward slash at end

kappa_list = [1]
degree_list = [1]
method_list = ['nonlocal_integral_eq']
method_to_kwargs = {
    'transmission': {
        'options_prefix': 'tr_',
    },
    'pml': {
        'pml_type': 'bdy_integral',
        'options_prefix': 'pml_',
    },
    'nonlocal_integral_eq': {
        'cl_ctx': cl_ctx,
        'queue': queue,
        'with_refinement': True,
    }
}
pml_options_prefix = ''
transmission_options_prefix = ''

# Use cache if have it?
use_cache = True

# Write over duplicate trials?
write_over_duplicate_trials = False

# Visualize solutions?
visualize = False

cache_file_name = "data/2d_hankel_trial.csv"


def get_fmm_order(kappa, h):
    """
        :arg kappa: The wave number
        :arg h: The maximum characteristic length of the mesh
    """
    return 6

# }}}


# Open cache file to get any previously computed results
print("Reading cache...")
try:
    in_file = open(cache_file_name)
    cache_reader = csv.DictReader(in_file)
    cache = {}

    for entry in cache_reader:

        output = {}
        for output_name in ['l2_relative_error', 'h1_relative_error', 'ndofs']:
            output[output_name] = entry[output_name]
            del entry[output_name]
        cache[frozenset(entry.items())] = output

    in_file.close()
except (OSError, IOError):
    cache = {}
print("Cache read in")

uncached_results = {}

if write_over_duplicate_trials:
    uncached_results = cache

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
# (some of these are for just pml, but we don't
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
mesh_h_vals = []
for filename in os.listdir(mesh_file_dir):
    basename, ext = os.path.splitext(filename)  # remove ext
    if ext == '.msh':
        meshes.append(fd.Mesh(mesh_file_dir + basename + ext))

        hstr = basename[3:]
        hstr = hstr.replace("%", ".")
        h = float(hstr)
        mesh_h_vals.append(h)

meshes.sort(key=lambda x: x.coordinates.dat.data.shape[0])
mesh_h_vals.sort(reverse=True)
print("Meshes Read in.")


# All the input parameters to a run
setup_info = {}
# Store error and functions
results = {}

iteration = 0
total_iter = len(meshes) * len(degree_list) * len(kappa_list) * len(method_list)


field_names = ('h', 'degree', 'kappa', 'method', 'fmm_order',
               'ndofs', 'l2_relative_error', 'h1_relative_error')
for mesh, mesh_h in zip(meshes, mesh_h_vals):
    setup_info['h'] = str(mesh_h)
    x, y = fd.SpatialCoordinate(mesh)

    for degree in degree_list:
        setup_info['degree'] = str(degree)

        for kappa in kappa_list:
            setup_info['kappa'] = str(kappa)
            true_sol_expr = fd.Constant(1j / 4) \
                * hankel_function(kappa * fd.sqrt(x**2 + y**2),
                                  n=hankel_cutoff)

            trial = {'mesh': mesh,
                     'degree': degree,
                     'true_sol_expr': true_sol_expr}

            for method in method_list:
                setup_info['method'] = method

                if method == 'nonlocal_integral_eq':
                    fmm_order = get_fmm_order(kappa, mesh_h)
                    setup_info['fmm_order'] = str(fmm_order)

                # Gets computed solution, prints and caches
                key = frozenset(setup_info.items())

                if not use_cache or key not in cache:
                    kwargs = method_to_kwargs[method]
                    true_sol, comp_sol = run_method(trial, method, kappa,
                                                    comp_sol_name=method
                                                    + " Computed Solution",
                                                    **kwargs)

                    uncached_results[key] = {}

                    l2_err = norms.l2_norm(true_sol - comp_sol, region=inner_region)
                    l2_true_sol_norm = norms.l2_norm(true_sol, region=inner_region)
                    l2_relative_error = l2_err / l2_true_sol_norm

                    h1_err = norms.h1_norm(true_sol - comp_sol, region=inner_region)
                    h1_true_sol_norm = norms.h1_norm(true_sol, region=inner_region)
                    h1_relative_error = h1_err / h1_true_sol_norm

                    uncached_results[key]['l2_relative_error'] = l2_relative_error
                    uncached_results[key]['h1_relative_error'] = h1_relative_error

                    ndofs = true_sol.dat.data.shape[0]
                    uncached_results[key]['ndofs'] = str(ndofs)

                else:
                    ndofs = cache[key]['ndofs']
                    l2_relative_error = cache[key]['l2_relative_error']
                    h1_relative_error = cache[key]['h1_relative_error']

                    if visualize:
                        fd.plot(comp_sol)
                        fd.plot(true_sol)
                        plt.show()

                iteration += 1
                print('iter %s / %s' % (iteration, total_iter))
                print('h:', mesh_h)
                print("ndofs:", ndofs)
                print("kappa:", kappa)
                print("method:", method)
                print('degree:', degree)
                if 'fmm_order' in setup_info:
                    c = 0.5
                    print('Epsilon= %.2f^(%d+1) = %f'
                          % (c, fmm_order, c**(fmm_order+1)))
                    del setup_info['fmm_order']

                print("L^2 Relative Err: ", l2_relative_error)
                print(" Relative Err: ", h1_relative_error)
                print()

        # write to cache if necessary
        if uncached_results:
            print("Writing to cache...")

            write_header = False
            if write_over_duplicate_trials:
                out_file = open(cache_file_name, 'w')
                write_header = True
            else:
                if not os.path.isfile(cache_file_name):
                    write_header = True
                out_file = open(cache_file_name, 'a')

            cache_writer = csv.DictWriter(out_file, field_names)

            if write_header:
                cache_writer.writeheader()

            # {{{ Move data to cache dictionary and append to file
            #     if not writing over duplicates
            for key in uncached_results:
                if key in cache and not write_over_duplicate_trials:
                    out_file.close()
                    raise ValueError('Duplicating trial, maybe set'
                                     ' write_over_duplicate_trials to *True*?')

                row = dict(key)
                for output in uncached_results[key]:
                    row[output] = uncached_results[key][output]

                if not write_over_duplicate_trials:
                    cache_writer.writerow(row)
                cache[key] = uncached_results[key]

            uncached_results = {}
            
            # }}}

            # {{{ Re-write all data if writing over duplicates

            if write_over_duplicate_trials:
                for key in cache:
                    row = dict(key)
                    for output in uncached_results[key]:
                        row[output] = uncached_results[key][output]
                    cache_writer.writerow(row)

            # }}}

            out_file.close()

            print("cache closed")
