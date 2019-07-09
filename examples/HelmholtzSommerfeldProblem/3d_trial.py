import os
import pyopencl as cl

cl_ctx = cl.create_some_context()
queue = cl.CommandQueue(cl_ctx)

# For WSL, all firedrake must be imported after pyopencl
import firedrake as fd
from methods.run_method import run_method
import pickle

# Trial settings
mesh_file_dir = "ball_in_cube/"  # NEED a forward slash at end
kappa_list = [3]
degree_list = [1]
method_list = ['nonlocal_integral_eq']
method_to_kwargs = {
    'transmission': {},
    'nonlocal_integral_eq': {
        'cl_ctx': cl_ctx,
        'queue': queue,
        'with_refinement': True,
        'epsilon': 0.20,
        'print_fmm_order': True}
    }

# Use cache if have it?
use_cache = False

cache_file_name = "3d_trial.pickle"
try:
    in_file = open(cache_file_name, 'rb')
    cache = pickle.load(in_file)
    in_file.close()
except (OSError, IOError):
    cache = {}

# Hankel approximation cutoff
hankel_cutoff = 25

inner_bdy_id = 2
outer_bdy_id = 1

# Set kwargs that don't expect user to change
# (NOTE: some of these are for just pml, but we don't
#  expect the user to want to change them
global_kwargs = {'scatterer_bdy_id': inner_bdy_id,
                 'outer_bdy_id': outer_bdy_id,
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

        hstr = basename[4:]
        hstr = hstr.replace("%", ".")
        h = float(hstr)
        mesh_h_vals.append(h)

meshes.sort(key=lambda x: x.coordinates.dat.data.shape[0])
print("Meshes Read in.")


def get_key(setup_info):
    sorted_keys = sorted(setup_info.keys())
    key = tuple([setup_info[key] for key in sorted_keys])
    return key


def relative_error(true_sol, comp_sol):
    true_sol_norm = fd.sqrt(fd.assemble(
        fd.inner(true_sol, true_sol) * fd.dx
        ))
    l2_err = fd.sqrt(fd.assemble(
        fd.inner(true_sol - comp_sol, true_sol - comp_sol) * fd.dx
        ))
    return l2_err / true_sol_norm


# All the input parameters to a run
setup_info = {}
# Store error and functions
results = {}

for mesh, mesh_h in zip(meshes, mesh_h_vals):
    setup_info['h'] = mesh_h
    x, y, z = fd.SpatialCoordinate(mesh)
    norm = fd.sqrt(x**2 + y**2 + z**2)

    for degree in degree_list:
        setup_info['degree'] = degree

        for kappa in kappa_list:
            setup_info['kappa'] = kappa
            true_sol_expr = fd.Constant(1 / (4*fd.pi)) / norm \
                * fd.exp(1j * kappa * norm)

            trial = {'mesh': mesh,
                     'degree': degree,
                     'true_sol_expr': true_sol_expr}

            for method in method_list:
                setup_info['method'] = method
                key = get_key(setup_info)

                if not use_cache or key not in cache:

                    kwargs = method_to_kwargs[method]
                    true_sol, comp_sol = run_method(trial, method, kappa,
                                                    comp_sol_name=method
                                                    + " Computed Solution",
                                                    **kwargs)
                    rel_err = relative_error(true_sol, comp_sol)

                    ndofs = true_sol.dat.data.shape[0]

                    # Store method
                    results[key] = (rel_err, true_sol, comp_sol)
                    cache[key] = (rel_err, ndofs)

                else:
                    rel_err, ndofs = cache[key]
                    results[key] = (rel_err, None, None)

                print("ndofs:", ndofs)
                print("kappa:", kappa)
                print("method:", method)
                print('degree:', degree)
                print("L^2 Relative Err: ", rel_err)
                print()

        # write to cache
        out_file = open(cache_file_name, 'wb')
        pickle.dump(cache, out_file)
        out_file.close()


import matplotlib.pyplot as plt
def make_plot(independent_var):

    assert independent_var in setup_info.keys()
    key_to_key = {key: key for key in setup_info}
    key_entries = get_key(key_to_key)  # tuple of what each entry in the key is

    const_vars = set()
    const_vars_str = ""

    for i, entry in enumerate(key_entries):
        if entry == independent_var:
            continue

        entry_vals = set()
        for key in results:
            entry_vals.add(key[i])

        if len(entry_vals) == 1:
            const_vars.add(entry)
            val, = list(entry_vals)
            const_vars_str += str(entry) + "=" + str(val) + "; "

    # Map all but independent_var to [(x_1, y_1), ..., (x_n, y_n)]
    new_results = {}
    for key in results:
        new_setup_info = {key_entry: entry for
                          key_entry, entry in zip(key_entries, key)}

        ind_value = new_setup_info[independent_var]
        del new_setup_info[independent_var]
        for const_var in const_vars:
            del new_setup_info[const_var]

        new_key = get_key(new_setup_info)

        rel_err = results[key][0]
        new_results.setdefault(new_key, []).append((ind_value, rel_err))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Relative Error vs. %s\n%s" % (independent_var, const_vars_str))
    for key in new_results:
        x, y = zip(*(new_results[key]))  # This actually unzips

        label = ""
        i = 0
        for entry in key_entries:
            if entry == independent_var or entry in const_vars:
                continue
            label += str(entry) + "=" + str(key[i]) + "; "
            i += 1

        ax.scatter(x, y, label=label)
        ax.set_xlabel(independent_var)
        ax.set_ylabel("Relative Error")
    ax.legend()


#make_plot('kappa')
#plt.show()