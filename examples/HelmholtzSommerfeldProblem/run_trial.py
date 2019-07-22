import os
import csv
import matplotlib.pyplot as plt
import pyopencl as cl

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

# For WSL, all firedrake must be imported after pyopencl
import firedrake as fd
import utils.norm_functions as norms
from methods import run_method

from firedrake.petsc import OptionsManager
from firedrake.solving_utils import KSPReasons
from utils.hankel_function import hankel_function
from mpi4py import MPI

import faulthandler
faulthandler.enable()

# {{{ Trial settings for user to modify

mesh_file_dir = "circle_in_square/"  # NEED a forward slash at end
mesh_dim = 2

kappa_list = [0.1, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0]
degree_list = [1]
method_list = ['pml', 'transmission', 'nonlocal_integral_eq']
method_to_kwargs = {
    'transmission': {
        'options_prefix': 'transmission',
        'solver_parameters': {'pc_type': 'lu',
                              'preonly': None,
                              'ksp_rtol': 1e-12,
                              },
    },
    'pml': {
        'pml_type': 'bdy_integral',
        'options_prefix': 'pml',
        'solver_parameters': {'pc_type': 'lu',
                              'preonly': None,
                              'ksp_rtol': 1e-12,
                              }
    },
    'nonlocal_integral_eq': {
        'cl_ctx': cl_ctx,
        'queue': queue,
        'options_prefix': 'nonlocal',
        'solver_parameters': {'pc_type': 'lu',
                              'ksp_rtol': 1e-12,
                              },
    }
}

# Use cache if have it?
use_cache = False

# Write over duplicate trials?
write_over_duplicate_trials = True

# min h, max h? Only use meshes with characterstic length in [min_h, max_h]
min_h = 0.125
max_h = None

# Visualize solutions?
visualize = False

def get_fmm_order(kappa, h):
    """
        :arg kappa: The wave number
        :arg h: The maximum characteristic length of the mesh
    """
    return 49

# }}}

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
comm = MPI.COMM_WORLD.Split(rank)

# Make sure not using pml if in 3d
if mesh_dim != 2 and 'pml' in method_list:
    raise ValueError("PML not implemented in 3d")


# Open cache file to get any previously computed results
print("Reading cache...")
cache_file_name = "data/" + mesh_file_dir[:-1] + '.csv'  # :-1 to take off slash
try:
    in_file = open(cache_file_name)
    cache_reader = csv.DictReader(in_file)
    cache = {}

    for entry in cache_reader:

        output = {}
        for output_name in ['L^2 Relative Error', 'H^1 Relative Error', 'ndofs',
                            'Iteration Number', 'Residual Norm', 'Converged Reason']:
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
if mesh_dim == 2:
    hankel_cutoff = None

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
elif mesh_dim == 3:
    hankel_cutoff = None

    inner_bdy_id = 2
    outer_bdy_id = 1
    inner_region = None
    pml_x_region = None
    pml_y_region = None
    pml_xy_region = None

    pml_x_min = None
    pml_x_max = None
    pml_y_min = None
    pml_y_max = None


def get_true_sol_expr(spatial_coord):
    if mesh_dim == 3:
        x, y, z = spatial_coord
        norm = fd.sqrt(x**2 + y**2 + z**2)
        return fd.Constant(1j / (4*fd.pi)) / norm * fd.exp(1j * kappa * norm)

    elif mesh_dim == 2:
        x, y = spatial_coord
        return fd.Constant(1j / 4) * hankel_function(kappa * fd.sqrt(x**2 + y**2),
                                                     n=hankel_cutoff)
    raise ValueError("Only meshes of dimension 2, 3 supported")


# Set kwargs that don't expect user to change
# (some of these are for just pml, but we don't
#  expect the user to want to change them
#
# The default solver parameters here are the defaults for
# a :class:`LinearVariationalSolver`, see
# https://www.firedrakeproject.org/solving-interface.html#id19
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
                 'solver_parameters': {'snes_type': 'ksponly',
                                       'ksp_type': 'gmres',
                                       'ksp_gmres_restart': 30,
                                       'ksp_rtol': 1.0e-7,
                                       'ksp_atol': 1.0e-50,
                                       'ksp_divtol': 1e4,
                                       'ksp_max_it': 10000,
                                       'pc_type': 'ilu'
                                       },
                 }

# Go ahead and make the file directory accurate
mesh_file_dir = 'meshes/' + mesh_file_dir

# Ready kwargs by defaulting any absent kwargs to the global ones
for mkey in method_to_kwargs:
    for gkey in global_kwargs:
        if gkey not in method_to_kwargs[mkey]:
            method_to_kwargs[mkey][gkey] = global_kwargs[gkey]


print("Preparing Mesh Names...")
mesh_names = []
mesh_h_vals = []
for filename in os.listdir(mesh_file_dir):
    basename, ext = os.path.splitext(filename)  # remove ext
    if ext == '.msh':
        mesh_names.append(mesh_file_dir + basename + ext)

        hstr = basename[3:]
        hstr = hstr.replace("%", ".")
        h = float(hstr)
        mesh_h_vals.append(h)

# Sort by h values
mesh_h_vals_and_names = zip(mesh_h_vals, mesh_names)
if min_h is not None:
    mesh_h_vals_and_names = [(h, n) for h, n in mesh_h_vals_and_names if h >= min_h]
if max_h is not None:
    mesh_h_vals_and_names = [(h, n) for h, n in mesh_h_vals_and_names if h <= max_h]

mesh_h_vals, mesh_names = zip(*sorted(mesh_h_vals_and_names, reverse=True))
print("Meshes Prepared.")

# {{{ Get setup options for each method
solver_params_list = []
for method in method_list:
    # Get the solver parameters
    solver_parameters = dict(global_kwargs.get('solver_parameters', {}))
    for k, v in method_to_kwargs[method].get('solver_parameters', {}).items():
        solver_parameters[k] = v

    options_prefix = method_to_kwargs[method].get('options_prefix', None)

    options_manager = OptionsManager(solver_parameters, options_prefix)
    options_manager.inserted_options()
    solver_params_list.append(options_manager.parameters)

# }}}


# All the input parameters to a run
setup_info = {}
# Store error and functions
results = {}

iteration = 0
total_iter = len(mesh_names) * len(degree_list) * len(kappa_list) * len(method_list)
plus_one = 0
if total_iter % size > rank:
    plus_one = 1
total_iter = total_iter // size + plus_one

field_names = ('h', 'degree', 'kappa', 'method',
               'pc_type', 'preonly', 'FMM Order', 'ndofs',
               'L^2 Relative Error', 'H^1 Relative Error', 'Iteration Number',
               'Residual Norm', 'Converged Reason', 'ksp_rtol', 'ksp_atol')
mesh = None
for mesh_name, mesh_h in zip(mesh_names, mesh_h_vals):
    setup_info['h'] = str(mesh_h)

    if mesh is not None:
        del mesh
        mesh = None

    for degree in degree_list:
        setup_info['degree'] = str(degree)

        for kappa in kappa_list:
            setup_info['kappa'] = str(float(kappa))
            true_sol_expr = None

            trial = {'mesh': mesh,
                     'degree': degree,
                     'true_sol_expr': true_sol_expr}

            for method, solver_params in zip(method_list, solver_params_list):
                if iteration % size != rank:
                    iteration += 1
                    continue
                setup_info['method'] = str(method)
                setup_info['pc_type'] = str(solver_params['pc_type'])
                setup_info['preonly'] = str('preonly' in solver_params)
                if 'preonly' in solver_params:
                    setup_info['ksp_rtol'] = ''
                    setup_info['ksp_atol'] = ''
                else:
                    setup_info['ksp_rtol'] = str(solver_params['ksp_rtol'])
                    setup_info['ksp_atol'] = str(solver_params['ksp_atol'])

                if method == 'nonlocal_integral_eq':
                    fmm_order = get_fmm_order(kappa, mesh_h)
                    setup_info['FMM Order'] = str(fmm_order)
                    method_to_kwargs[method]['FMM Order'] = fmm_order
                else:
                    setup_info['FMM Order'] = ''

                # Gets computed solution, prints and caches
                key = frozenset(setup_info.items())

                if not use_cache or key not in cache:
                    # {{{  Read in mesh if haven't already
                    if mesh is None:
                        print("\nReading Mesh...")
                        mesh = fd.Mesh(mesh_name, comm=comm)
                        spatial_coord = fd.SpatialCoordinate(mesh)
                        trial['mesh'] = mesh
                        print("Mesh Read in.\n")

                    if true_sol_expr is None:
                        true_sol_expr = get_true_sol_expr(spatial_coord)
                        trial['true_sol_expr'] = true_sol_expr

                    # }}}

                    kwargs = method_to_kwargs[method]
                    true_sol, comp_sol, ksp = run_method.run_method(
                        trial, method, kappa,
                        comp_sol_name=method + " Computed Solution", **kwargs)

                    uncached_results[key] = {}

                    l2_err = norms.l2_norm(true_sol - comp_sol, region=inner_region)
                    l2_true_sol_norm = norms.l2_norm(true_sol, region=inner_region)
                    l2_relative_error = l2_err / l2_true_sol_norm

                    h1_err = norms.h1_norm(true_sol - comp_sol, region=inner_region)
                    h1_true_sol_norm = norms.h1_norm(true_sol, region=inner_region)
                    h1_relative_error = h1_err / h1_true_sol_norm

                    uncached_results[key]['L^2 Relative Error'] = l2_relative_error
                    uncached_results[key]['H^1 Relative Error'] = h1_relative_error

                    ndofs = true_sol.dat.data.shape[0]
                    uncached_results[key]['ndofs'] = str(ndofs)
                    uncached_results[key]['Iteration Number'] = \
                        ksp.getIterationNumber()
                    uncached_results[key]['Residual Norm'] = \
                        ksp.getResidualNorm()
                    uncached_results[key]['Converged Reason'] = \
                        KSPReasons[ksp.getConvergedReason()]
                    if str(uncached_results[key]['Converged Reason']) \
                            == 'KSP_DIVERGED_NANORINF':
                        print("\nKSP_DIVERGED_NANORINF\n")

                    if visualize:
                        fd.plot(comp_sol)
                        fd.plot(true_sol)
                        plt.show()

                else:
                    ndofs = cache[key]['ndofs']
                    l2_relative_error = cache[key]['L^2 Relative Error']
                    h1_relative_error = cache[key]['H^1 Relative Error']

                iter_to_print = iteration // size + 1
                iteration += 1
                print('iter:   %s / %s on rank %s process (%s processes total)' %
                      (iter_to_print, total_iter, rank, size))
                print('h:     ', mesh_h)
                print("ndofs: ", ndofs)
                print("kappa: ", kappa)
                print("method:", method)
                print('degree:', degree)
                if setup_info['method'] == 'nonlocal_integral_eq':
                    c = 0.5
                    print('Epsilon= %.2f^(%d+1) = %e'
                          % (c, fmm_order, c**(fmm_order+1)))

                print("L^2 Relative Err: ", l2_relative_error)
                print("H^1 Relative Err: ", h1_relative_error)
                print()


# Go ahead and free any data while waiting for other processes
del mesh
del run_method.prepared_trials
del run_method.memoized_objects

if rank != 0:
    MPI.COMM_WORLD.send(uncached_results, dest=0)
else:
    for i in range(1, size):
        # Get uncached results from process i
        i_uncached_results = MPI.COMM_WORLD.recv(source=i)
        for key, val in i_uncached_results.items():
            uncached_results[key] = val

# write to cache if necessary for rank 0
if uncached_results and rank == 0:
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
            for output in cache[key]:
                row[output] = cache[key][output]
            cache_writer.writerow(row)

    # }}}

    out_file.close()

    print("cache closed")