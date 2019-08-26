import os
import csv
import matplotlib.pyplot as plt
import pyopencl as cl

# For WSL, all firedrake must be imported after pyopencl
from firedrake import sqrt, Constant, pi, exp, Mesh, SpatialCoordinate, \
    plot
import utils.norm_functions as norms
from utils.to_2nd_order import to_2nd_order
from methods import run_method

from firedrake.petsc import OptionsManager, PETSc
from firedrake.solving_utils import KSPReasons
from multiprocessing.pool import Pool
from utils.hankel_function import hankel_function

import faulthandler
faulthandler.enable()

# {{{ Trial settings for user to modify

mesh_file_dir = "circle_in_square/"  # NEED a forward slash at end
mesh_dim = 2
num_processes = None  # None defaults to os.cpu_count()

kappa_list = [0.1, 1.0, 3.0, 5.0]
degree_list = [1]
method_list = ['transmission', 'pml', 'nonlocal']
method_to_kwargs = {
    'transmission': {
        'options_prefix': 'transmission',
        'solver_parameters': {'pc_type': 'lu',
                              'ksp_type': 'preonly',
                              },
    },
    'pml': {
        'pml_type': 'bdy_integral',
        'options_prefix': 'pml',
        'solver_parameters': {'pc_type': 'lu',
                              'ksp_type': 'preonly',
                              }
    },
    'nonlocal': {
        'options_prefix': 'nonlocal',
        'solver_parameters': {'pc_type': 'lu',
                              'ksp_compute_singularvalues': None,
                              'ksp_rtol': 1e-12,
                              },
    }
}

# Use cache if have it?
use_cache = False

# Write over duplicate trials?
write_over_duplicate_trials = True

# min h, max h? Only use meshes with characterstic length in [min_h, max_h]
min_h = None
max_h = None

# Print trials as they are completed?
print_trials = True

# Visualize solutions?
visualize = False

# use 2nd order mesh?
use_2nd_order = False


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

if visualize and mesh_dim == 3:
    raise ValueError("Visualization not implemented in 3d")


# Open cache file to get any previously computed results
print("Reading cache...")
cache_file_name = "data/" + mesh_file_dir[:-1] + '.csv'  # :-1 to take off slash
try:
    in_file = open(cache_file_name)
    cache_reader = csv.DictReader(in_file)
    cache = {}

    for entry in cache_reader:

        output = {}
        for output_name in ['L2 Error', 'H1 Error', 'ndofs',
                            'Iteration Number', 'Residual Norm', 'Converged Reason',
                            'Min Extreme Singular Value',
                            'Max Extreme Singular Value']:
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


print("Reading in Mesh names...")
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
print("Meshes prepared.")

# {{{ Get setup options for each method
for method in method_list:
    # Get the solver parameters
    solver_parameters = dict(global_kwargs.get('solver_parameters', {}))
    for k, v in method_to_kwargs[method].get('solver_parameters', {}).items():
        solver_parameters[k] = v

    options_prefix = method_to_kwargs[method].get('options_prefix', None)

    options_manager = OptionsManager(solver_parameters, options_prefix)
    options_manager.inserted_options()
    method_to_kwargs[method]['solver_parameters'] = options_manager.parameters
# }}}


# All the input parameters to a run
setup_info = {'2nd Order': str(use_2nd_order)}
# Store error and functions
results = {}

iteration = 0
total_iter = len(mesh_names) * len(degree_list) * len(kappa_list) * len(method_list)


current_mesh_name = None
mesh = None
def run_trial(trial_id):
    """
    (key, output), or *None* if use_cache is *True*
    and already have this result stored
    """
    # {{{  Get indexes into lists
    mesh_ndx = trial_id % len(mesh_names)
    trial_id //= len(mesh_names)
    mesh_name = mesh_names[mesh_ndx]
    # If new mesh, delete old mesh and read in new one
    global current_mesh_name, mesh
    if current_mesh_name != mesh_name:
        del mesh
        mesh = Mesh(mesh_name)
        if use_2nd_order:
            mesh = to_2nd_order(mesh)
        current_mesh_name = mesh_name

    mesh_h = mesh_h_vals[mesh_ndx]

    degree_ndx = trial_id % len(degree_list)
    trial_id //= len(degree_list)
    degree = degree_list[degree_ndx]

    kappa_ndx = trial_id % len(kappa_list)
    trial_id //= len(kappa_list)
    kappa = kappa_list[kappa_ndx]

    method_ndx = trial_id % len(method_list)
    trial_id //= len(method_list)
    method = method_list[method_ndx]
    solver_params = method_to_kwargs[method]['solver_parameters']

    # Make sure this is a valid iteration index
    assert trial_id == 0
    # }}}

    kwargs = method_to_kwargs[method]
    # {{{ Create key holding data of this trial run:

    setup_info['h'] = str(mesh_h)
    setup_info['degree'] = str(degree)
    setup_info['kappa'] = str(kappa)
    setup_info['method'] = str(method)

    setup_info['pc_type'] = str(solver_params['pc_type'])
    if solver_params['ksp_type'] == 'preonly':
        setup_info['ksp_rtol'] = ''
        setup_info['ksp_atol'] = ''
    else:
        setup_info['ksp_rtol'] = str(solver_params['ksp_rtol'])
        setup_info['ksp_atol'] = str(solver_params['ksp_atol'])

    if method == 'nonlocal':
        fmm_order = get_fmm_order(kappa, mesh_h)
        setup_info['FMM Order'] = str(fmm_order)
        kwargs['FMM Order'] = fmm_order
    else:
        setup_info['FMM Order'] = ''

    # Add gamma/beta to setup_info if there, else make sure
    # it's recorded as absent in special_key
    for special_key in ['gamma', 'beta']:
        if special_key in solver_params:
            setup_info[special_key] = solver_params[special_key]
        else:
            setup_info[special_key] = ''

    key = frozenset(setup_info.items())
    if key in cache and use_cache:
        return None

    trial = {'mesh': mesh,
             'degree': degree,
             'true_sol_expr': get_true_sol_expr(SpatialCoordinate(mesh),
                                                kappa)}

    # {{{ Solve problem and evaluate error
    output = {}

    true_sol, comp_sol, snes_or_ksp = \
        run_method.run_method(trial, method, kappa,
                              comp_sol_name=method + " Computed Solution", **kwargs)

    if isinstance(snes_or_ksp, PETSc.SNES):
        ksp = snes_or_ksp.getKSP()
    elif isinstance(snes_or_ksp, PETSc.KSP):
        ksp = snes_or_ksp
    else:
        raise ValueError("snes_or_ksp must be of type PETSc.SNES or"
                         " PETSc.KSP")

    l2_err = norms.l2_norm(true_sol - comp_sol, region=inner_region)
    h1_err = norms.h1_norm(true_sol - comp_sol, region=inner_region)

    # }}}

    # Store err in output and return

    output['L2 Error'] = l2_err
    output['H1 Error'] = h1_err

    ndofs = true_sol.dat.data.shape[0]
    output['ndofs'] = str(ndofs)
    if solver_params['ksp_type'] != 'preonly':
        output['Iteration Number'] = ksp.getIterationNumber()
    output['Residual Norm'] = ksp.getResidualNorm()
    output['Converged Reason'] = KSPReasons[ksp.getConvergedReason()]

    if solver_params['ksp_type'] == 'gmres':
        emin, emax = ksp.computeExtremeSingularValues()
        output['Min Extreme Singular Value'] = emin
        output['Max Extreme Singular Value'] = emax

    if visualize:
        plot(comp_sol)
        plot(true_sol)
        plt.show()

    if print_trials:
        for name, val in sorted(key):
            if val != '':
                print('{0: <9}: {1}'.format(name, val))
        for name, val in sorted(output.items()):
            print('{0: <18}: {1}'.format(name, val))

    return key, output


def initializer(method_to_kwargs):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    method_to_kwargs['nonlocal']['queue'] = queue


# Run pool, map setup info to output info
with Pool(processes=num_processes, initializer=initializer,
          initargs=(method_to_kwargs,)) as pool:
    print("computing")
    new_results = pool.map(run_trial, range(total_iter))

new_results = filter(lambda x: x is not None, new_results)
uncached_results = {**uncached_results, **dict(new_results)}

field_names = ('h', 'degree', 'kappa', 'method',
               'pc_type', 'FMM Order', 'ndofs', '2nd order',
               'L2 Error', 'H1 Error', 'Iteration Number',
               'gamma', 'beta', 'ksp_type',
               'Residual Norm', 'Converged Reason', 'ksp_rtol', 'ksp_atol',
               'Min Extreme Singular Value', 'Max Extreme Singular Value')
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
