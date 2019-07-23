import os
import csv
import matplotlib.pyplot as plt
import pyopencl as cl

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

# For WSL, all firedrake must be imported after pyopencl
from firedrake import sqrt, Constant, pi, exp, Mesh, SpatialCoordinate, \
    plot
import utils.norm_functions as norms
from methods import run_method

from firedrake.petsc import OptionsManager
from firedrake.solving_utils import KSPReasons
from mpi4py.futures import MPIPoolExecutor as Pool
from utils.hankel_function import hankel_function

import faulthandler
faulthandler.enable()

# {{{ Trial settings for user to modify

mesh_file_dir = "circle_in_square/"  # NEED a forward slash at end
mesh_dim = 2
num_processes = 2  # None defaults to os.cpu_count()

kappa_list = [0.1, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0]
degree_list = [1]
#method_list = ['pml', 'transmission', 'nonlocal_integral_eq']
method_list = ['nonlocal_integral_eq']
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
min_h = 0.5
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


def get_true_sol_expr(spatial_coord, kappa):
    if mesh_dim == 3:
        x, y, z = spatial_coord
        norm = sqrt(x**2 + y**2 + z**2)
        return Constant(1j / (4*pi)) / norm * exp(1j * kappa * norm)

    elif mesh_dim == 2:
        x, y = spatial_coord
        return Constant(1j / 4) * hankel_function(kappa * sqrt(x**2 + y**2),
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


print("Reading in Meshes...")
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
# Read in the meshes
meshes = [Mesh(name) for name in mesh_names]
print("Meshes read in.")

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


def get_key_args_kwargs(iter_num):
    """
    Returns (key, fntn) Where fntn will return
    (true_sol, comp_sol, ksp), or *None* if use_cache is *True*
    and already have this result stored
    """
    # {{{  Get indexes into lists
    mesh_ndx = iter_num % len(meshes)
    iter_num //= len(meshes)

    degree_ndx = iter_num % len(degree_list)
    iter_num //= len(degree_list)

    kappa_ndx = iter_num % len(kappa_list)
    iter_num //= len(kappa_list)

    method_ndx = iter_num % len(method_list)
    iter_num //= len(method_list)

    # Make sure this is a valid iteration index
    assert iter_num == 0
    # }}}

    kwargs = method_to_kwargs[method_list[method_ndx]]
    # {{{ Create key holding data of this trial run:

    setup_info['h'] = str(mesh_h_vals[mesh_ndx])
    setup_info['degree'] = str(degree_list[degree_ndx])
    setup_info['kappa'] = str(kappa_list[kappa_ndx])
    setup_info['method'] = str(method_list[method_ndx])

    solver_params = solver_params_list[method_ndx]

    setup_info['pc_type'] = str(solver_params['pc_type'])
    setup_info['preonly'] = str('preonly' in solver_params)
    if 'preonly' in solver_params:
        setup_info['ksp_rtol'] = ''
        setup_info['ksp_atol'] = ''
    else:
        setup_info['ksp_rtol'] = str(solver_params['ksp_rtol'])
        setup_info['ksp_atol'] = str(solver_params['ksp_atol'])

    if method_list[method_ndx] == 'nonlocal_integral_eq':
        fmm_order = get_fmm_order(kappa_list[kappa_ndx], mesh_h_vals[mesh_ndx])
        setup_info['FMM Order'] = str(fmm_order)
        kwargs['FMM Order'] = fmm_order
    else:
        setup_info['FMM Order'] = ''

    key = frozenset(setup_info.items())
    if key in cache and use_cache:
        return None

    mesh = meshes[mesh_ndx]
    trial = {'mesh': mesh,
             'degree': degree_list[degree_ndx],
             'true_sol_expr': get_true_sol_expr(SpatialCoordinate(mesh),
                                                kappa_list[kappa_ndx])}

    # Precomputation
    kwargs['no_run'] = True
    global cl_ctx
    if cl_ctx is None:
        cl_ctx = cl.create_some_context()
        queue = cl.CommandQueue(cl_ctx)

    kwargs['cl_ctx'] = cl_ctx
    kwargs['queue'] = queue
    run_method.run_method(trial, method_list[method_ndx], kappa_list[kappa_ndx],
                          **kwargs)
    del kwargs['no_run']

    return key, (trial, method_list[method_ndx], kappa_list[kappa_ndx],
                 'True Solution', method + ' Computed Solution'), kwargs
    # }}}


print("Creating Function Spaces and Converters (and other args)...")
"""
pool_args = []
for i in range(total_iter):
    print("Constructing function spaces, converters, etc. %d/%d" % (i+1, total_iter))
    if get_key_args_kwargs(i) is not None:
        pool_args.append(i)
"""


def get_output(i):
    output = {}

    key, args, kwargs = get_key_args_kwargs(i)
    global cl_ctx
    if cl_ctx is None:
        cl_ctx = cl.create_some_context()
        queue = cl.CommandQueue(cl_ctx)

    kwargs['cl_ctx'] = cl_ctx
    kwargs['queue'] = queue
    true_sol, comp_sol, ksp = run_method.run_method(*args, **kwargs)

    l2_err = norms.l2_norm(true_sol - comp_sol, region=inner_region)
    l2_true_sol_norm = norms.l2_norm(true_sol, region=inner_region)
    l2_relative_error = l2_err / l2_true_sol_norm

    h1_err = norms.h1_norm(true_sol - comp_sol, region=inner_region)
    h1_true_sol_norm = norms.h1_norm(true_sol, region=inner_region)
    h1_relative_error = h1_err / h1_true_sol_norm

    output['L^2 Relative Error'] = l2_relative_error
    output['H^1 Relative Error'] = h1_relative_error

    ndofs = true_sol.dat.data.shape[0]
    output['ndofs'] = str(ndofs)
    output['Iteration Number'] = ksp.getIterationNumber()
    output['Residual Norm'] = ksp.getResidualNorm()
    output['Converged Reason'] = KSPReasons[ksp.getConvergedReason()]
    if str(uncached_results[key]['Converged Reason']) \
            == 'KSP_DIVERGED_NANORINF':
        print("\nKSP_DIVERGED_NANORINF\n")

    if visualize:
        plot(comp_sol)
        plot(true_sol)
        plt.show()

    return key, output


"""
if num_processes is None:
    num_processes = os.cpu_count()
print("Running %s / %s requested trials, on %s processes,"
      " remaining already stored" %
      (len(pool_args), total_iter, num_processes))
"""

# Run pool, map setup info to output info
if __name__ == '__main__':
    with Pool(max_workers=8) as pool:
        pool_args = pool.map(get_key_args_kwargs, range(total_iter))
        uncached_results = dict(zip*(pool.map(get_output, pool_args)))


field_names = ('h', 'degree', 'kappa', 'method',
               'pc_type', 'preonly', 'FMM Order', 'ndofs',
               'L^2 Relative Error', 'H^1 Relative Error', 'Iteration Number',
               'Residual Norm', 'Converged Reason', 'ksp_rtol', 'ksp_atol')
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
            for output in cache[key]:
                row[output] = cache[key][output]
            cache_writer.writerow(row)

    # }}}

    out_file.close()

    print("cache closed")
